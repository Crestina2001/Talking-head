import cv2
import numpy as np
import os
import face_detection
from os import path
import torch

# Ensure that the face detection model is present.
if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
    raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth before running this script!')

# Initialize face alignment using the first GPU (or the CPU if no GPU is available).
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

# Path to the video file.
vfile = 'HDTF_25fps_2/RD_Radio1_000/00000.mp4'

# Prepare the directory to save the detected faces.
vidname = os.path.basename(vfile).split('.')[0]
dirname = 'detected_faces'
fulldir = path.join(dirname, vidname)
os.makedirs(fulldir, exist_ok=True)

# Process the video file.
video_stream = cv2.VideoCapture(vfile)
batch_size = 16  # Adjust the batch size to your needs.

frames = []
while True:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break
    frames.append(frame)

# Run the face detection in batches.
batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
i = -1
for fb in batches:
    preds = fa.get_detections_for_batch(np.asarray(fb))

    for j, f in enumerate(preds):
        i += 1
        if f is None:
            continue

        x1, y1, x2, y2 = f
        cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])

print(f'Finished processing {vfile}. Detected faces are saved in {fulldir}.')
