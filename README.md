# This is the repo for our SLT work:

# Improving End-to-end Sign Language Translation with Adaptive Video Representation Enhanced Transformer

This repo contains the training and evaluation code for the sign language translation (SLT) on RWTH-PHOENIX-Weather 2014 T (PH14T) dataset.

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
### Step 1: Prepare the visual features:
* For Gloss-based features of PH14T, please download the pre-trained visual features from [SLT](https://github.com/neccam/slt) project.

### Step 2: Pack the visual features:
Following the [SLT](https://github.com/neccam/slt) project to compressed the features by gzip and pickle. Then, put them into `AVRET/data/`

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

## Others

* For gloss-free features and codes of PH14T, please see [GFSLT-VLP-AVRET-PH14T](https://github.com/LzDddd/GFSLT-VLP-AVRET-PH14T).
* For gloss-based features and codes of CSL-Daily, please see [TwoStreamNetwork](https://github.com/LzDddd/TwoStreamNetwork-AVRET).
* For gloss-free features and codes of CSL-Daily, please see [GFSLT-VLP-AVRET-CSL](https://github.com/LzDddd/GFSLT-VLP-AVRET-CSL).
