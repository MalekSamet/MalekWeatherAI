# Thesis Repository 

## Contribution
This work aims for augmenting a dataset (SANPO) in terms of weather conditions and e-scooters. It provides training, inference and evaluation of the whole work.

## Overview
This repository serves as a guide to effectively use and manage a suite of interconnected projects. Each project focuses on a specific aspect of the workflow, and this document provides clear instructions for setup, usage, and integration.

---

## Features
This repository enables users to perform the following tasks:
1. **Train ALDM on SANPO**
2. **Inference ALDM on SANPO**
3. **Run Image Captioning w/ BLIP**
4. **Sanpo dataset re-annotation**
5. **Inference Inpaint_Anything**
6. **Segmentation Model for Evaluation**

---

## Submodules and Conda Environments
This repository consists of the following submodules, each with its corresponding Conda environment (All environments are in the folder "/environment" and they have also copies in their corresponding repositories):

1. **[repo1]https://github.com/MalekSamet/ALDM_Thesis.git** 
   - Conda Environment: `environment_ALDM.yml`

2. **[repo2]https://github.com/MalekSamet/BLIP_Thesis.git** 
   - Conda Environment: `environment_ALDM.yml`

3. **[repo3]https://github.com/MalekSamet/InpaintAnything_Thesis.git** 
   - Conda Environment: `environment_inpaint.yml`

4. **[repo4]https://github.com/MalekSamet/OneFormer_Thesis.git** 
   - Conda Environment: `environment_oneformer.yml`

5. **[repo5]https://github.com/MalekSamet/SegmentationModel_Thesis.git** 
   - Conda Environment: `env_repo5`

---

## Setup Instructions
To get started, clone this repository along with its submodules:
For HTTPS:

```bash
git clone --recurse-submodules https://github.com/MalekSamet/ThesisProject.git
cd MalekThesis
```
For SSH:

```bash
git clone --recurse-submodules git@github.com:MalekSamet/ThesisProject.git
cd MalekThesis
```

## Features Usage
### 1.Train ALDM on SANPO

First, we need environment_ALDM.yml:
```bash
cd ALDM_Thesis
conda env create -f environment_ALDM.yml
conda activate ALDM
```
Before starting the training, ensure that the dataset has the following structure in the folder dataset/:
**SANPO Original:**
dataset │ ├── sanpo │ ├── images │ ├── train │ └── val │ └── annotations ├── train └── val

**SANPO Edit (3 extra classes for snow, bench & billboard):**
dataset │ ├── sanpo │ ├── images │ ├── train │ └── val │ └── processed_annotations ├── train └── val

To train the model:
```cd ADLM_Laajim
python train_ALDM.py  --sanpo_mode original --call_BLIP true
```
sanpo_mode: 'original' or 'edit'
call_BLIP: true for generation of image captions, false otherwise

To set more hyperparameters, check the scripts train_ALDM_sanpo_original.py or train_ALDM_sanpo.edit.py

You can find corresponding encoders, decoders and weight initialization: (Paste these under /pretrained) 
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/decoder_epoch_50_30cls.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/decoder_epoch_50_31cls.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/decoder_epoch_50_33cls.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/decoder_epoch_50_34cls.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/decoder_epoch_50_151cls.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/checkpoint/control_seg_enc_scratch.ckpt
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/pretrained/resnet101-imagenet.pth

### 2.Inference ALDM on SANPO
```cd ALDM_Thesis
python inference_sanpo.py  --sanpo_mode original --inference_mode manual --checkpoint /path/to/model/weigths --save_dir /path/dir/to/save/outputs --folder_path /path/input/images/dir
```
- argument sanpo_mode: 'original' and 'edit' as explained in the previous section. It could be also 'cs', to run a pre-trained cityscapes model, and maps the sanpo input to cityscapes.
- argument inference_mode: 'manual' allows to paste the mask input path manually in the console, as well as the prompt, n_prompt and seed within a loop until the user stops the program. For 'folder', the model infers all the masks within one directory "--folder_path". The user still gives the prompts and the seed in the console.

You can find corresponding model weights: (for original sanpo, paste under /checkpoint/old_model, for edit sanpo, paste under /checkpoint/new_model, for cs, paste under /checkpoint)
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/checkpoint/new_model/epoch=60-step=45599.ckpt
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/checkpoint/old_model/epoch=60-step=45599.ckpt
	/home/malek_ma/Desktop/ROOT_FOLDER/ALDM/checkpoint/cityscapes_step9.ckpt


### 3.Run Image Captioning with BLIP
Generates a dict of image captions in a json file
```cd BLIP_Thesis
python run_BLIP.py --image_folder /path/to/images/dir --output_json_path /path/output/json
```
### 4.SANPO dataset re-annotation
```
cd OneFormer_Thesis
python sanpo_edit_mask_processing.py --images_dir /path/to/images/ --mask_dir /path/to/annotations --output_dir /path/to/output/directory --model_weights /path/to/model/weights
```

You can find the model weights:
	/home/malek_ma/Desktop/ROOT_FOLDER/OneFormer/checkpoint/250_16_swin_l_oneformer_mapillary_300k.pth
### 5.Inference Inpaint_Anything
First, create and activate the environment:
```
cd InpaintAnything_Thesis
conda env create -f environment_inpaint.yml
conda activate inpaint
```
The e-scooter insertion is performed on 3 steps:
	1. Setting position, size and orientation of the rectngular mask on the background image
	2. Clicking on a pixel in the white mask
	3. Generate output image

To insert e-scooter on an image, run this script:
```
cd InpaintAnything_Thesis
python inference_image.py --input_image /path/to/image/ --text_prompt prompt_of_object_to_insert --output_dir /path/to/output/directory --sam_ckpt /path/to/model/weigths --rectangle_width 0.65 --rectangle_height 1
```
To run the insertion for several images in a directory, run this script:
```
cd InpaintAnything_Thesis
python inference_folder.py --input_dir /path/to/images/directory --text_prompt prompt_of_object_to_insert --output_dir /path/to/output/directory --sam_ckpt /path/to/model/weigths --rectangle_width 0.65 --rectangle_height 1
```
You can find the model weights in: (best ckpt must be pasted under /pretrained_models/big-lama/models)
	/home/malek_ma/Desktop/ROOT_FOLDER/Inpaint-Anything/pretrained_models/sam_vit_h_4b8939.pth
	/home/malek_ma/Desktop/ROOT_FOLDER/Inpaint-Anything/pretrained_models/big-lama/models/best.ckpt
	
	
### 6. Segmentation Model for Evaluation
	
