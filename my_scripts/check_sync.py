import os
from pydub import AudioSegment

# Function to get audio length in seconds
def get_audio_length(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    return len(audio) / 1000.0

# Base directory containing preprocessed data
base_dir = 'preprocessed'

# Check if the base directory exists
if not os.path.exists(base_dir):
    print(f"The directory {base_dir} does not exist.")
else:
    print(f"Checking synchronization in {base_dir}...")

# Iterate over all folders within the base directory
for list_folder in os.listdir(base_dir):
    list_folder_path = os.path.join(base_dir, list_folder)
    # Ensure it's a directory
    if os.path.isdir(list_folder_path):
        # Iterate over all video ID folders within the list folder
        for video_id_folder in os.listdir(list_folder_path):
            video_id_path = os.path.join(list_folder_path, video_id_folder)
            # Ensure it's a directory
            if os.path.isdir(video_id_path):
                # Print the current video ID being checked
                print(f"Checking video ID: {video_id_folder} in folder {list_folder}...")
                # Get the list of frame files for each video ID folder
                frame_files = [f for f in os.listdir(video_id_path) if f.endswith('.jpg')]
                num_frames = len(frame_files)
                print(f"Found {num_frames} frames in video ID {video_id_folder}...")
                # Path to the audio file
                audio_path = os.path.join(video_id_path, 'audio.wav')
                # Check if the audio file exists
                if not os.path.isfile(audio_path):
                    print(f"Audio file missing for video ID {video_id_folder}")
                else:
                    # Get the audio length in seconds
                    audio_length = get_audio_length(audio_path)
                    # Calculate the FPS
                    fps = num_frames / audio_length if audio_length > 0 else 0
                    # Print the FPS calculation regardless of the value
                    print(f'Video ID: {video_id_folder}, Calculated FPS: {fps:.2f}')
                    
                    # Define an expected FPS value and check for deviations
                    expected_fps = 25  # Adjust as per your expected FPS
                    if abs(fps - expected_fps) > 1:  # Allow for some margin of error
                        print(f'Warning: Significant deviation detected in video ID {video_id_folder}')
