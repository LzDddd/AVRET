# This is a temporary repo for our SLT work:

# Improving End-to-end Sign Language Translation with Adaptive Video Representation Enhanced Transformer

This repo contains the training and evaluation code for the sign language translation (SLT).

The code is based on [SLT](https://github.com/neccam/slt) but modified to realize SLT without gloss in training. Due to the code structure issues, we will continue to update the repo.

## Environment
```shell
git clone https://github.com/LzDddd/AVRET.git
cd AVRET
conda create -n avret_env python==3.7
conda activate avret_env
pip install -r requirements.txt
```

## Datasets
### Step 1: Download the raw data
* [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
* [CSL-Daily](https://ustc-slr.github.io/datasets/2021_csl_daily/)

### Step 2: Prepare the visual features:
* Download the pre-trained visual features from [SLT](https://github.com/neccam/slt).
* (Simple version) Or, using the [Efficientnet-b0](https://github.com/lukemelas/EfficientNet-PyTorch) instead of the Encoder in [VideoMoCo](https://github.com/tinapan-pt/VideoMoCo) and pre-train the Discriminator by VideoMoCo framework. Then, using the pre-trained Efficientnet-b0 to extract visual features and compressed it by gzip (like [SLT](https://github.com/neccam/slt)). 
## Usage
Firstly, make sure the data folder is as follows:
```shell
AVRET
└── data
    ├── phoenix14t.pami0.dev
    ├── phoenix14t.pami0.test
    └── phoenix14t.pami0.train
```

Pre-training

`python -m signjoey vlp_pretrain configs/sign_vlp.yaml`

Training

`python -m signjoey train configs/sign.yaml`

Evaluation

`python -m signjoey test configs/sign.yaml  --ckpt <ckpt_path> --output_path <results_output_path>`

## TODO
- [X] *Initial code release.*
- [ ] *Reformat the code*

## 
