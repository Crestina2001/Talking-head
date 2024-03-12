import os
import glob
from moviepy.editor import VideoFileClip

def has_audio(file_path):
    try:
        video = VideoFileClip(file_path)
        return video.audio is not None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def update_text_files(silent_videos, filelists_path):
    for txt_file in ['train.txt', 'val.txt', 'test.txt']:
        file_path = os.path.join(filelists_path, txt_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                if not any(silent_video in line for silent_video in silent_videos):
                    file.write(line)

def move_folders(silent_videos, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for video in silent_videos:
        source_folder = os.path.join(source_dir, video)
        target_folder = os.path.join(target_dir, video)

        if os.path.exists(source_folder):
            os.rename(source_folder, target_folder)

# Main script
video_dir = 'HDTF_25fps'
silent_videos = []
for file_path in glob.glob(os.path.join(video_dir, '*.mp4')):
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    if not has_audio(file_path):
        silent_videos.append(video_name)

# Update text files
filelists_path = 'filelists'
update_text_files(silent_videos, filelists_path)

# Move corresponding folders
source_dir = 'preprocessed'
target_dir = 'cached_videos'
move_folders(silent_videos, source_dir, target_dir)
