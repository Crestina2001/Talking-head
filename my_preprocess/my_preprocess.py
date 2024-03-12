import os
import subprocess
from glob import glob
from moviepy.editor import VideoFileClip

# Your long video files directory
data_root = 'HDTF_25fps'
# Where you want to save the short clips, structured like LRS2
preprocessed_root = 'HDTF_25fps_2'

# Define the length of each clip you want to split the videos into, in seconds
clip_length = 5  # for 5 seconds long clips

# Create the preprocessed_root directory if it doesn't exist
os.makedirs(preprocessed_root, exist_ok=True)

def split_video(video_path, start_time, clip_length, output_path):
    """
    This function uses ffmpeg to split the video.
    """
    end_time = start_time + clip_length
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(clip_length),
        '-c', 'copy',
        output_path,
        '-y'  # Overwrite without asking
    ]
    result = subprocess.run(command)
    if result.returncode != 0:  # Non-zero return code indicates an error
        print(f"Error processing {output_path}")

def process_videos(data_root, preprocessed_root, clip_length):
    # Find all .mp4 files in the data_root directory
    video_files = glob(os.path.join(data_root, '*.mp4'))

    for video_file in video_files:
        try:
            # Load the video file
            clip = VideoFileClip(video_file)
            video_duration = clip.duration
        except Exception as e:
            print(f"Error loading video {video_file}: {e}")
            continue  # Skip this file and continue with the next

        # Calculate the number of segments to split
        num_clips = int(video_duration // clip_length)
        
        # Get the base name of the video file
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        
        # Create a subfolder for each original video file
        subfolder_path = os.path.join(preprocessed_root, base_name)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Split the video into segments
        for i in range(num_clips):
            start_time = i * clip_length
            output_filename = f"{i:05d}.mp4"  # Format the filename like '00001.mp4'
            output_path = os.path.join(subfolder_path, output_filename)
            split_video(video_file, start_time, clip_length, output_path)

# Run the preprocessing
process_videos(data_root, preprocessed_root, clip_length)
