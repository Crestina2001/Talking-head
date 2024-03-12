import os
import random

# Set the path to your preprocessed dataset directory
preprocessed_dir = 'HDTF_25fps_2'

# Define the paths for the train.txt and val.txt files
train_filelist_path = 'filelists/train.txt'
val_filelist_path = 'filelists/val.txt'

# Ratio of the validation set to the total number of videos
val_ratio = 0.1  # For example, 10% of the data will be used as a validation set

# List to hold all the video names
all_videos = []

# Walk through the preprocessed dataset directory and collect all video names
for subdir, dirs, files in os.walk(preprocessed_dir):
    # We assume each subdir in the root is a separate video directory
    if subdir != preprocessed_dir:  # Ignore the root directory
        relative_subdir = os.path.relpath(subdir, preprocessed_dir)
        all_videos.append(relative_subdir.replace(os.path.sep, '/'))  # Replace os-specific path separators

# Shuffle the list to ensure random distribution
random.shuffle(all_videos)

# Calculate the number of validation samples
num_val_samples = int(len(all_videos) * val_ratio)

# Split the videos into training and validation sets
val_videos = all_videos[:num_val_samples]
train_videos = all_videos[num_val_samples:]

# Write the video names to train.txt
with open(train_filelist_path, 'w') as f_train:
    for video in train_videos:
        f_train.write(f"{video}\n")

# Write the video names to val.txt
with open(val_filelist_path, 'w') as f_val:
    for video in val_videos:
        f_val.write(f"{video}\n")

print(f"Training on {len(train_videos)} videos, validating on {len(val_videos)} videos.")
