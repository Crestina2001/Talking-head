This repo is based upon: https://github.com/Rudrabha/Wav2Lip

## Demo

![](./demo/001.mp4)

## Installation

```
conda env create -f environment.yml
conda activate fairseq_env
```

- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.
- ffmpeg: `sudo apt-get install ffmpeg`

## Data Processing

### 1. Prepare dataset

I used the data from https://github.com/MRzzm/HDTF?tab=readme-ov-file. You could use any dataset you love.

The FPS shall be 25 !!

Original data shall looks like(I am writing like this to show that the naming of the mp4 files does not matter):

```
folder
	|---332.mp4
	|---dsf.mp4
	|---23r.mp4
	|...
```

### 2. My_Preprocessing

Use the script: my_preprocess/my_preprocess.py

Adjust the relative path by revising this line:

```
data_root = 'HDTF_25fps' # change to the folder of your own dataset
```

Then run the script:

```
python my_preprocess.py
```

The result shall look like:

```
folder
	|---332(name of the sub-folder)
		|---00000.mp4
		|---00001.mp4
		|---00002.mp4
		|...
	|---dsf
		|---00000.mp4
		|---00001.mp4
		|---00002.mp4
		|...
	|...
```

Each clip of video shall have a length of 5 seconds, except the last one. No need to pad it.

### 3. Train-Val-Test List Generation

Generate train-val-test lists(results will be put under the folder named 'filelists')

Copy the my_preprocess/train_list_generator.py into the same director as your processed dataset(or you could simply adjust the relative path to make it right). The processed dataset refers to the result of the second step. Revise the following line of code in the script to make it work:

```
preprocessed_dir = 'HDTF_25fps_2' # change to the name of your folder
```

Then run:

```
python train_list_generator.py
```

### 4. Wav2Lip preprocessing

```
python preprocess.py --data_root PATH-TO-STEP3-RESULT --preprocessed_root preprocessed/
```

the 'preprocessed' refers to name of the output folder.

The result shall look like:

```
preprocessed
	|---332
		|---00000
			|---0.jpg
			|---1.jpg
			|---2.jpg
			|...
			|audio.wav
		|---00001
			|---0.jpg
			|---1.jpg
			|---2.jpg
			|...
			|audio.wav
		|...
	|---dsf
		|---00000
			|---0.jpg
			|---1.jpg
			|---2.jpg
			|...
			|audio.wav
		|---00001
			|---0.jpg
			|---1.jpg
			|---2.jpg
			|...
			|audio.wav
		|...
	|...
```

Each image is the face part of each video frame.

### 5. Generate audio embedding with pretrained models

Download the model from [hubert](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md), or [contentvec](https://github.com/auspicious3000/contentvec)

For contentvec, use this guy: ContentVec_legacy_500

For HuBERT, use HuBERT Base (~95M params)(without finetuning)

Run the script(put in my_preprocess/contentvec_preprocess.py, move it to the correct directory or adjust the relative path):

```
python contentvec_preprocess.py --data_root preprocessed --contentvec_checkpoint_path YOUR-CONTENTVEC-OR-HUBERT-PATH
```

## Train the SyncNet

copy the my_scripts/contentvec_syncnet_train.py into the main directory

Run the following code(suppose your data is stored in the folder named preprocessed):

```
python contentvec_syncnet_train.py
```

## Train the Wav2Lip

copy the my_scripts/contentvec_train.py into the main directory

Run the following code(suppose your data is stored in the folder named preprocessed):

```
python contentvec_train.py
```

