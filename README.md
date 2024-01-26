# This is a temporary repo for our SLT work:

# Improving End-to-end Sign Language Translation with Adaptive Video Representation Enhanced Transformer

This repo contains the training and evaluation code for the sign language translation (SLT) of gloss-based RWTH-PHOENIX-Weather 2014 T.

The code is based on [SLT](https://github.com/neccam/slt) but modified to realize SLT without gloss in training.

## Environment
```shell
git clone https://github.com/LzDddd/AVRET.git
cd AVRET
conda create -n avret_env python==3.7
conda activate avret_env
pip install -r requirements.txt
```

## Datasets
### Step 1: Download the raw data:
* [RWTH-PHOENIX-Weather 2014 T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

### Step 2: Prepare the visual features:
Download the pre-trained visual features from [SLT](https://github.com/neccam/slt) project.

### Step 3: Pack the visual features:
Following the [SLT](https://github.com/neccam/slt) project to compressed the features by gzip and pickle. Then, put them into `./data/`

## Usage
Firstly, make sure the data folder is as follows:
```shell
AVRET
└── data
    ├── phoenix14t.pami0.dev
    ├── phoenix14t.pami0.test
    └── phoenix14t.pami0.train
```

Training command

`python -m signjoey train configs/sign.yaml`

Evaluation command

`python -m signjoey test configs/sign.yaml  --ckpt <ckpt_path> --output_path <results_output_path>`

## TODO
- [X] *Initial code release.*
- [ ] *Reformat the code.*
