from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import contentvec_Wav2Lip as Wav2Lip
import platform
import itertools
from collections import deque
import torch
import torchaudio
import numpy as np
import fairseq
from hparams import hparams

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('--contentvec_model', required = True,
					help='contentvecModel')


parser.add_argument('--face_landmarks_detector_path', default='inference_models/face_landmarker_v2_with_blendshapes.task',
						type=str, help='Path to face landmarks detector')
parser.add_argument('--with_face_mask', action='store_true', default = True,
						help='Blend output into original frame using a face mask rather than directly blending the face box. This prevents a lower resolution square artifact around lower face')

parser.add_argument('--audio_overlapping', default=0,
						type=int, help='contentvec overlapping')
args = parser.parse_args()
args.img_size = 96



def process_audio(wavpath):
    # Load the content vector model
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.contentvec_model])
    contentvec_model = models[0]
    contentvec_model.eval()

    # Load and preprocess the waveform
    waveform, sample_rate = torchaudio.load(wavpath, format='wav')
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = 16000
    waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

    num_samples = waveform.size(1)
    if num_samples < 32000:
        pad_length = 32000 - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    elif num_samples > 250000:
        waveform = waveform[:, :250000]

    waveform = waveform.squeeze().unsqueeze(0)

    # Extract features
    with torch.inference_mode():
        if isinstance(contentvec_model, torch.nn.DataParallel):
            features, _ = contentvec_model.module.extract_features(waveform)
        else:
            features, _ = contentvec_model.extract_features(waveform)
    wav = audio.load_wav(wavpath, hparams.sample_rate)
    old_mel = audio.melspectrogram(wav).T

    # Resampling to old melspectrogram dimension
    dim = old_mel.shape[0]
    features = torch.nn.functional.interpolate(features.transpose(1, 2), size=(dim,), mode='linear', align_corners=True)
    features = features.transpose(1, 2).squeeze(0)

    # Convert to numpy if needed
    if features.is_cuda:
        features = features.cpu().numpy()
    else:
        features = features.numpy()

    return features.transpose()


if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def datagen(frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def consume(iterator):
    "Consume an iterator, return the number of items consumed."
    counter = itertools.count()
    deque(itertools.zip_longest(iterator, counter), maxlen=0)
    return next(counter)

def face_mask_from_image(image, face_landmarks_detector):
	"""
	Calculate face mask from image. This is done by

	Args:
		image: numpy array of an image
		face_landmarks_detector: mediapipa face landmarks detector
	Returns:
		A uint8 numpy array with the same height and width of the input image, containing a binary mask of the face in the image
	"""
	# initialize mask
	mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

	# detect face landmarks
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
	detection = face_landmarks_detector.detect(mp_image)

	if len(detection.face_landmarks) == 0:
		# no face detected - set mask to all of the image
		mask[:] = 1
		return mask

	# extract landmarks coordinates
	face_coords = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]] for lm in detection.face_landmarks[0]])

	# calculate convex hull from face coordinates
	convex_hull = cv2.convexHull(face_coords.astype(np.float32))

	# apply convex hull to mask
	mask = cv2.fillPoly(mask, pts=[convex_hull.squeeze().astype(np.int32)], color=1)

	# Apply Gaussian blur to the edges of the mask
	# You can adjust the kernel size (e.g., (15, 15)) and sigma (e.g., 10) to control the blur intensity
	kernel_size = (35, 35)
	sigma = 20
	smooth_mask = cv2.GaussianBlur(mask, kernel_size, sigma)
	return smooth_mask

def split_audio_into_segments(audio_path, segment_length=5, overlap=args.audio_overlapping, target_dir='generated_faces'):
    """
    Splits an audio file into overlapping segments and saves them in the target directory.
    Returns a list of paths to the audio segments.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the total length of the audio in seconds
    waveform, sample_rate = torchaudio.load(audio_path)
    total_length = waveform.shape[1] / sample_rate

    segment_paths = []
    start = 0
    while start < total_length:
        end = min(start + segment_length + overlap, total_length)
        segment_file = os.path.join(target_dir, f'segment_{start:.2f}_{end:.2f}.wav')
        command = f'ffmpeg -y -i "{audio_path}" -ss {start} -to {end} -c copy "{segment_file}"'
        subprocess.call(command, shell=True)
        segment_paths.append(segment_file)

        start += segment_length - overlap  # Move start for the next segment back by the overlap duration

    return segment_paths

def process_all_segments(segment_paths, overlap_duration = args.audio_overlapping):
    all_features = []
    overlap_samples = int(overlap_duration * 80)

    for i, segment_path in enumerate(segment_paths):
        segment_features = process_audio(segment_path)

        if i > 0:
            if overlap_samples==0:
                segment_features = segment_features
            # For all but the first segment, remove the first 'overlap_samples' of features
            segment_features = segment_features[:, overlap_samples:]

        if i < len(segment_paths) - 1:
            # For all but the last segment, keep all but the last 'overlap_samples' of features
            if overlap_samples==0:
                all_features.append(segment_features)
            else:
                all_features.append(segment_features[:, :-overlap_samples])
        else:
            # For the last segment, keep all features
            all_features.append(segment_features)

    return np.concatenate(all_features, axis=1)

def main():
	os.makedirs('generated_faces', exist_ok=True)
	if not os.path.isfile(args.face):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps

	else:
		video_stream = cv2.VideoCapture(args.face)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)


	print ("Number of frames available for inference: "+str(len(full_frames)))

	if not args.audio.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

		subprocess.call(command, shell=True)
		args.audio = 'temp/temp.wav'

	#wav = audio.load_wav(args.audio, 16000)
	#mel = process_audio(args.audio)
	audio_segments = split_audio_into_segments(args.audio)
	mel = process_all_segments(audio_segments)
	print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)

	# Count for unique image names
	img_count = 0
	if args.with_face_mask:
		BaseOptions = mp.tasks.BaseOptions
		FaceLandmarker = mp.tasks.vision.FaceLandmarker
		FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
		VisionRunningMode = mp.tasks.vision.RunningMode

		options = FaceLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=args.face_landmarks_detector_path),
			running_mode=VisionRunningMode.IMAGE)

		face_landmarks_detector = FaceLandmarker.create_from_options(options)


	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(args.checkpoint_path)
			print ("Model loaded")

			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi', 
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			if args.with_face_mask:
				mask = face_mask_from_image(p, face_landmarks_detector)
				f[y1:y2, x1:x2] = f[y1:y2, x1:x2] * (1 - mask[..., None]) + p * mask[..., None]
			else:
				f[y1:y2, x1:x2] = p
			out.write(f)
			# Write the face to the face_out VideoWriter
			# Save the face as an image
			cv2.imwrite(os.path.join('generated_faces', f'face_{img_count}.jpg'), p)
        	# Increment the image count
			img_count += 1



	out.release()

	command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
	subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
	main()
