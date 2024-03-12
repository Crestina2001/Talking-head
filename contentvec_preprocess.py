import os, argparse
import glob
import fairseq.checkpoint_utils
from tqdm import tqdm
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import torchaudio
import torch.nn.functional as F
from hparams import hparams
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default = 'preprocessed/', type=str)

parser.add_argument('--contentvec_checkpoint_path', help='Load the contentvec path', default = 'inference_models/hubert_base_ls960.pt', type=str)

args = parser.parse_args()

class contentvec_cls:
    def __init__(self):
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.contentvec_checkpoint_path])
        #models, cfg, task = load_model_ensemble_and_task_from_hf_hub(hubert_model_name)
        self.contentvec_model = models[0]
        self.contentvec_model.eval()
    def process_wav_file(self, wavpath):
        # load the model if it exists
        root_path = wavpath.replace('audio.wav', '')

        waveform, sample_rate = torchaudio.load(wavpath, format='wav')

        num_channels = waveform.size(0)
        if num_channels > 1:
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
        #print(waveform.shape)


        # Extract features
        with torch.inference_mode():
            if isinstance(self.contentvec_model, torch.nn.DataParallel):
                features, _ = self.contentvec_model.module.extract_features(waveform)
            else:
                features, _ = self.contentvec_model.extract_features(waveform)

        wav = audio.load_wav(wavpath, hparams.sample_rate)

        old_mel = audio.melspectrogram(wav).T

        # resampling to old melspectrogram dimension
        dim = old_mel.shape[0]
        features = F.interpolate(features.transpose(1, 2), \
        size=(dim,), mode='linear', align_corners=True)

        features = features.transpose(1, 2).squeeze(0)
        # transform the features into numpy
        if features.is_cuda:
            features = features.cpu().numpy()
        else:
            features = features.numpy()
        #print(features.shape)
        feature_path = root_path + 'hb_feature.npy'
        np.save(feature_path, features)
        print(f"feature saved to {feature_path}")

        #old_mel_path = root_path + 'old_mel.npy'
        #np.save(old_mel_path, old_mel)
        #print(f"old mels saved to {old_mel_path}")


def traverse_and_process(root_folder):
    model = contentvec_cls()
    # Traverse the root folder
    for root, dirs, files in os.walk(root_folder):
        # Check if the current folder is a sub-sub-folder
        # (i.e., it has no sub-folders within it)
        if not dirs:
            # Use glob to find the .wav file in this folder
            wav_files = glob.glob(os.path.join(root, '*.wav'))
            # Process each found .wav file
            for wav_file in wav_files:
                model.process_wav_file(wav_file)

def delete_npy_files(directory):
    # Construct the pattern to match all .npy files
    pattern = os.path.join(directory, '**', '*.npy')

    # Use glob to find all .npy files, with recursive search
    npy_files = glob.glob(pattern, recursive=True)

    for file_path in npy_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

#delete_npy_files(args.data_root)
# Replace 'your_root_folder_path' with the path of your root folder
traverse_and_process(args.data_root)
