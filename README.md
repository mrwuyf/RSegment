## Introduction
RSegment is a toolbox of remote sensing images segmentation based on [GeoSeg](https://github.com/WangLibo1995/GeoSeg), which aims to help related personnel establishing a comprehensive remote sensing seamtic segmentation workflow.

## Folder Structure
Prepare the following folders to organize this repo:
```none  
RSegment  
├── RSegment (code)  
├── pretrain_weights (pretrained weights of backbones, such as vit, swin, etc)  
├── model_weights (save the model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (save the masks predicted by models)  
├── lightning_logs (CSV format training logs)
├── rs_process(tools for process rs images)
├── data  
│   ├── LoveDA  
│   │   ├── Train  
│   │   │   ├── Urban  
│   │   │   │   ├── images_png (original images)  
│   │   │   │   ├── masks_png (original masks)  
│   │   │   │   ├── masks_png_convert (converted masks used for training)  
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)  
│   │   │   ├── Rural  
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert  
│   │   │   │   ├── masks_png_convert_rgb  
│   │   ├── Val (the same with Train)  
│   │   ├── Test  
│   │   ├── train_val (Merge Train and Val)   
│   ├── vaihingen  
│   │   ├── train_images (original)  
│   │   ├── train_masks (original)  
│   │   ├── test_images (original)  
│   │   ├── test_masks (original)  
│   │   ├── test_masks_eroded (original)  
│   │   ├── train (processed)  
│   │   ├── test (processed)  
│   ├── potsdam (the same with vaihingen)
```

## Preparation
Create python enviroment as follows(for windows platform, if you use linux platform, please ignore arosics installation and find another ways to install gdal):
``` shell
conda create -n rseg python=3.8
conda activate rseg
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge 'arosics>=1.3.0'
pip install -r RSegment/requirements.txt
```
## Data Preparation

#### Vaihingen
Generate training set
```shell 
python RSegment/tools/vaihingen_nopad_split.py \  
--img-dir "data/vaihingen/train_images" \  
--mask-dir "data/vaihingen/train_masks" \  
--output-img-dir "data/vaihingen/train/images_1024" \  
--output-mask-dir "data/vaihingen/train/masks_1024" \  
--mode "train" --split-size 1024 --stride 512 
```
Generate testing set
```shell
python RSegment/tools/vaihingen_nopad_split.py \  
--img-dir "data/vaihingen/test_images" \  
--mask-dir "data/vaihingen/test_masks_eroded" \  
--output-img-dir "data/vaihingen/test/images_1024" \  
--output-mask-dir "data/vaihingen/test/masks_1024" \  
--mode "val" --split-size 1024 --stride 1024 \  
--eroded  
```
Generate masks for visualization
```shell
python RSegment/tools/vaihingen_nopad_split.py \  
--img-dir "data/vaihingen/test_images" \  
--mask-dir "data/vaihingen/test_masks" \  
--output-img-dir "data/vaihingen/test/images_1024" \  
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \  
--mode "val" --split-size 1024 --stride 1024 \  
--gt  
```
#### Potsdam
Generate training set
```shell
python RSegment/tools/potsdam_nopad_split.py \  
--img-dir "data/potsdam/train_images" \  
--mask-dir "data/potsdam/train_masks" \  
--output-img-dir "data/potsdam/train/images_1024" \  
--output-mask-dir "data/potsdam/train/masks_1024" \  
--mode "train" --split-size 1024 --stride 1024 --rgb-image
```
Generate testing set
```shell
python RSegment/tools/potsdam_nopad_split.py \  
--img-dir "data/potsdam/test_images" \  
--mask-dir "data/potsdam/test_masks_eroded" \  
--output-img-dir "data/potsdam/test/images_1024" \  
--output-mask-dir "data/potsdam/test/masks_1024" \  
--mode "val" --split-size 1024 --stride 1024 \  
--eroded --rgb-image
```
Generate masks for visualization
```shell
python RSegment/tools/potsdam_nopad_split.py \  
--img-dir "data/potsdam/test_images" \  
--mask-dir "data/potsdam/test_masks" \  
--output-img-dir "data/potsdam/test/images_1024" \  
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \  
--mode "val" --split-size 1024 --stride 1024 \  
--gt --rgb-image
```
#### LoveDA
```shell
python RSegment/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert  
python RSegment/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert  
python RSegment/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert  
python RSegment/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```

## Training and Testing
**Training**
```shell
python RSegment/train_supervision.py -c RSegment/config/vaihingen/unetfomer.py
```
**Vaihingen Testing**
```shell
python RSegment/vaihingen_test.py -c RSegment/config/vaihingen/unet.py -o fig_results/vaihingen/unet --rgb -t "d4"
```
**Potsdam Testing**
```shell
python RSegment/potsdam_test.py -c RSegment/config/potsdam/unet.py -o fig_results/vaihingen/unet --rgb -t "d4"
```
**LoveDA Testing**
```shell
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/unet.py -o fig_results/loveda/unet -t "d4" 
```
## Training Your Own Data
#### Training Step
First, build your own dataset. For remote sensing images like Gaofen and Sentinel, which are typically 16-bit images, you need to stretch them to 8-bit and select three bands for training. If you want to use multi-band training, read the images using GDAL.
Use Script in \RSegment\rs_process\makedataset to build dataset\
**stretch_image.py** for 16 to 8\
**crop.py** for croping image\
**irrgb2rgb.py** for changing bands\
**dataug.py** for data augmentation such as random flip, random rotate\
**removezero.py** for removing images which masks are all 0\
**splitdataset.py** for spliting tra set, val set and test set\
if you use 4 or more bands, you can read images as follows
```python
gdal.AllRegister()  
dataset_rs = gdal.Open(img_name)  
im_width = dataset_rs.RasterXSize  
im_height = dataset_rs.RasterYSize  
im_bands = dataset_rs.RasterCount  
img = dataset_rs.ReadAsArray(0, 0, im_width, im_height)  
img = np.transpose(img, (1, 2, 0))
```

Second, create your own dataset class by following the structure of the vaihingen_dataset in the dataset folder, then prepare the corresponding configuration file.

With these steps completed, you can proceed to start training and testing.

#### Inference Step
Change inference data folder in config
```shell
python GeoSeg/inference.py -c GeoSeg/config/siluan/unet.py -o fig_results/siluan/unet -t "d4" 
```
Then, use scripts in \RSegment\rs_process\postprocess to mosaic patch\
**crop.py** for croping image to inference\
**removezero.py** for removing images are all 0\
**mosaic.py** for mosaic patch

## Acknowledgement  
  
- [pytorch lightning](https://www.pytorchlightning.ai/)  
- [timm](https://github.com/rwightman/pytorch-image-models)  
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)  
- [ttach](https://github.com/qubvel/ttach)  
- [catalyst](https://github.com/catalyst-team/catalyst)  
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
