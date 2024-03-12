import os
import subprocess

def run_inference(base_folder):
    # Create a 'results' folder in the base folder if it doesn't exist
    results_folder = os.path.join(base_folder, "results")
    os.makedirs(results_folder, exist_ok=True)

    # List all video files in the base folder
    video_files = [f for f in os.listdir(base_folder) if f.endswith('.mp4')]
    for video_file in video_files:
        # Construct the corresponding audio file name
        audio_file = video_file.replace('.mp4', '.wav')

        # Check if the corresponding audio file exists
        if audio_file in os.listdir(base_folder):
            # Define full paths for video and audio files
            video_path = os.path.join(base_folder, video_file)
            audio_path = os.path.join(base_folder, audio_file)

            # Define the output file path in the 'results' folder
            output_file = os.path.join(results_folder, video_file)

            # Run the command
            run_command(video_path, audio_path, output_file)

def run_command(video_file, audio_file, output_file):
    cmd = [
        "python", "my_inference.py",
        "--checkpoint_path", "inference_models/con001_con002.pth",
        "--face", video_file,
        "--audio", audio_file,
        "--contentvec_model", "inference_models/checkpoint_best_legacy_500.pt",
        "--outfile", output_file
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    base_folder = "inference_data/dataset4"
    run_inference(base_folder)
