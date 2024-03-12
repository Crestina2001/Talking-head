import cv2
import tempfile
import torch
import dlib
import numpy as np
import skvideo.io
import fairseq
from fairseq import checkpoint_utils, tasks, utils
from av_hubert_models.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
import av_hubert_models.utils as avhubert_utils
from argparse import Namespace
from tqdm import tqdm



class AVHuBERTFeatureExtractor:
    def __init__(self, face_predictor_path, mean_face_path, ckpt_path, user_dir):
        self.face_predictor_path = face_predictor_path
        self.mean_face_path = mean_face_path
        self.ckpt_path = ckpt_path
        self.user_dir = user_dir

    
    def preprocess_video(self, input_video_path, output_video_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_predictor_path)
        STD_SIZE = (256, 256)
        mean_face_landmarks = np.load(self.mean_face_path)
        stablePntsIDs = [33, 36, 39, 42, 45]
        videogen = skvideo.io.vread(input_video_path)
        frames = np.array([frame for frame in videogen])
        landmarks = []
        for frame in tqdm(frames, desc="Processing Frames"):
            landmark = self.detect_landmark(frame, detector, predictor)
            landmarks.append(landmark)
        preprocessed_landmarks = landmarks_interpolate(landmarks)
        rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                          window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
        write_video_ffmpeg(rois, output_video_path, "ffmpeg")  # Adjust path to ffmpeg if needed

    def detect_landmark(self, image, detector, predictor):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rects = detector(gray, 1)
        coords = None
        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            coords = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def extract_visual_feature(self, video_path):
        utils.import_user_module(Namespace(user_dir=self.user_dir))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([self.ckpt_path])
        transform = avhubert_utils.Compose([
            avhubert_utils.Normalize(0.0, 255.0),
            avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
            avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)])
        frames = avhubert_utils.load_video(video_path)
        frames = transform(frames)
        frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).cuda()
        model = models[0]
        if hasattr(models[0], 'decoder'):
            model = models[0].encoder.w2v_model
        model.cuda()
        model.eval()
        with torch.no_grad():
            feature, _ = model.extract_finetune(source={'video': frames, 'audio': None}, padding_mask=None, output_layer=None)
            feature = feature.squeeze(dim=0)
        return feature

    def process_video(self, video_path):
        # Temporary path for processed video
        processed_video_path = tempfile.mktemp(suffix='.mp4')
        
        # Preprocess the video
        self.preprocess_video(video_path, processed_video_path)
        
        # Extract features
        features = self.extract_visual_feature(processed_video_path)
        
        return features

'''
# Instantiate the AVHuBERTFeatureExtractor class
face_predictor_path = "av_hubert_models/facial_landmark_detection.dat"
mean_face_path = "av_hubert_models/mean_face_landmarks.npy"
ckpt_path = "av_hubert_models/av_hubert.pt"
user_dir = "av_hubert_models"

av_hubert_extractor = AVHuBERTFeatureExtractor(face_predictor_path, mean_face_path, ckpt_path, user_dir)

# Path to the video you want to process
video_path = "video.mp4"

# Process the video and extract features
extracted_features = av_hubert_extractor.process_video(video_path)

# Print or use the extracted features
print("Extracted Features:", extracted_features)
'''
