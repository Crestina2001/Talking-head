import os
from glob import glob
from sklearn.model_selection import train_test_split

def prepare_data_lists(base_path, output_folder='filelists'):
    # Find all subdirectories containing frame sequences
    frame_dirs = glob(os.path.join(base_path, '*', '*'))

    # Split the data into training, validation, and testing sets
    train_dirs, test_dirs = train_test_split(frame_dirs, test_size=0.2, random_state=42)
    test_dirs, val_dirs = train_test_split(test_dirs, test_size=0.5, random_state=42)

    # Check if the output folder exists, if not, create it
    os.makedirs(output_folder, exist_ok=True)

    # Write the relative paths to the files
    for file_name, dataset in zip(["train.txt", "test.txt", "val.txt"], [train_dirs, test_dirs, val_dirs]):
        with open(os.path.join(output_folder, file_name), 'w', encoding='utf-8') as f:
            for dir_path in dataset:
                # Write the path relative to the base_path
                rel_path = os.path.relpath(dir_path, base_path)
                f.write(f"{rel_path}\n")

# Call the function with the path to your dataset
prepare_data_lists('preprocessed')
