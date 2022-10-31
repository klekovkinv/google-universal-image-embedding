# google-universal-image-embedding
Code for the kaggle competition 
[Google Universal Image Embedding](https://www.kaggle.com/competitions/google-universal-image-embedding).  
Solution for 35th place out of 1022 teams (top 4%).  

In this competition, the developed models are expected to 
retrieve relevant database images to a given query image 
(ie, the model should retrieve database images containing 
the same object as the query).  

#### ENV
* Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz  
* NVIDIA GeForce RTX 2070  
* Ubuntu 18.04.6 LTS  
* Driver Version: 470.42.01  
* CUDA Version: 11.4  
* CUDNN 8.2.4  

#### DEPENDENCIES
1. `conda create -n ggl python=3.7`  
1. `conda activate ggl`
1. `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`  
1. `pip install -r requirements.txt` 

#### USED DATASETS
* [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) (competition dataset)  
* [Products-10K](https://products-10k.github.io/) 
* [Google Landmark Recognition 2021](https://www.kaggle.com/competitions/landmark-recognition-2021) (competition dataset)  
* [RP2K: A Large-Scale Retail Product Dataset](https://www.pinlandata.com/rp2k_dataset/)  

#### DATASETS PREPARATION
Download datasets and put to `src/sources`.  

Run:  
* `python src/prepare_datasets/prepare_imagenet.py --input data/imagenet/train --output data/dataset --samples 55`  
* `python src/prepare_datasets/prepare_landmarks.py --input data/sources/landmark-recognition-2021 --output data/dataset --samples 55 --num-classes 10000`  
* `python src/prepare_datasets/prepare_products.py --input data/sources/data/sources/products10k --output data/dataset --samples 55 --num-classes 5000`  
* `python src/prepare_datasets/prepare_rp2k.py --input data/sources/data/sources/rp2k/train --output data/dataset --samples 55 --num-classes 1850`  

#### EMBEDDINGS EXTRACTION
```
python src/extract_embeddings.py \
--input data/dataset \
--clip-output data/embeddings/clip \
--open-clip-output data/embeddings/open_clip
```

#### TRAIN
```
python src/train.py \
--embeddings-1 data/embeddings/open_clip \
--embeddings-2 data/embeddings/clip \
--saving-folder models/ \
--num-classes 18450
```

#### INFERENCE
Inference code available in the [Competition Notebook](https://www.kaggle.com/code/klekovkin/35th-place-2xclip/notebook?scriptVersionId=107293192).  
