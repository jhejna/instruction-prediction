# Improving Long-Horizon Imitation Through Language Prediction

This is the official code repository for *Improving Long-Horizon Imitation Through Language Prediction* by [Donald (Joey) Hejna](https://jhejna.github.io), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), and [Lerrel Pinto](https://www.lerrelpinto.com/).

This repository includes easy to use code to reproduce the main results of the paper, which can be found [here](https://openreview.net/pdf?id=1Z3h4rCLvo-).

If you use this repository, please cite our work:
```
@inproceedings{
anonymous2022improving,
title={Improving Long-Horizon Imitation Through Language Prediction},
author={Anonymous},
booktitle={Submitted to The Tenth International Conference on Learning Representations },
year={2022},
url={https://openreview.net/forum?id=1Z3h4rCLvo-},
note={under review}
}
```

## Installation
All commands assume the user is in the repository directory.

1. Create the conda environment using the provided environment files. We recommend using GPU: `conda env create -f environment_gpu.yaml`. Then activate the conda environment.
2. Install the babyAI package as included in the repository: `cd babyai; pip install -e .`.
3. Install the mazebase package as included in the repo: `cd mazebase; pip install -e .`.
4. Install the langauge_prediction package: `pip install -e .`.

## Usage

### Generating Datasets
Datasets for the experiments can be generated using scripts from this repository.

To create the BabyAI datasets for BossLevel used in the paper, run the following command:
```
python scripts/create_babyai_dataset.py --env BabyAI-BossLevel-v0 --dataset-type traj traj_contrastive --max-mission-len 42 --skip 3 --episodes 50000 --valid-episodes 2500 --path datasets/BabyAIBossLevel_l42_50k --jobs 10
```
This will create two datasets, one with 'next' images used for unsupervised objectives and one without. The one without can be used in cases where memory is limited. The command also launches 10 parallel jobs, each collecting 5000 demos. The number of jobs can easily be reduced. Datasets for the other BabyAI levels can be created by modifying this command.

To create the dataset for the crafting environment, you will first need to download the raw data from [this repository](https://github.com/valeriechen/ask-your-humans). Download the raw json dataset. Then run the following command:
```
python scripts/create_crafting_dataset.py --input-path path/to/json/file --output-path datasets/crafting
```

### Training Models
To train a model, first update the corresponding configuration file in the `configs` folder with the path to the created dataset (should be fine if you ran the command listed above) and the desired parameters. Then run the following command:
```
python scripts/train.py --config configs/path/to/config --save-path path/to/output
```

Here are some example experiments from the paper:
1. BabyAI with Language Prediction and ATC:
```
python scripts/train.py --config configs/babyai/dt.yaml --save-path output/babyai/50k_lang0.7_unsup0.7
```
2. The Crafting Environment with Language Prediction and ATC:
```
python scripts/train.py --config configs/ayh/vit.yaml --save-path output/ayh/dataset0.4_lang0.25_unsup0.25
```
Note that the first time you run the ask your human experiments the GloVe embeddings will be downloaded by torchtext to `.vector_cache`.

The configs are self-explanatory and can be used to easily create the other experiments from the paper. If there is interest, I can also add the experiment sweeper to this repo that will run all the experiments at once.

### Monitoring Jobs
Jobs automatically output tensorboard logs to the specified location. You can view them using tensorboard.


### Testing Models
The final models are tested on held out levels. Results are logged during training, but can also be computed at the end using the following commands. Note that its important to match the options to those in the provided configuration files, or results may be inconsistent. The script will evaluate all models in a given folder. 
Here is the evaluation command for BabyAI:
```
python scripts/test.py --path path/to/model --best --num-ep 500 --eval-mode
```
Here is the evaluation command for Crafting on only unseen three-step levels. A list of all environment configurations for crafting can be found in `language_prediction/envs/mazebase.py
```
python scripts/test.py --path path/to/model --best --num-ep 500 --override env_kwargs.config unseen_3only
```
