# Paper Title

This is the official code repository for *paper title* by ...

This repository includes easy to use code to reproduce the main results of the paper.

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
This will create two datasets, one with 'next' images used for unsupervised objectives and one without. The one without can be used in cases where memory is limited. The command also launches 10 parallel jobs, each collecting 5000 demos. The number of jobs can easily be reduced. 

To create the dataset for the crafting environment, you will first need to download the raw data from [this repository](https://github.com/valeriechen/ask-your-humans). Download the raw json dataset. Then run the following command:
```
python scripts/create_crafting_dataset.py --input-path path/to/json/file --output-path datasets/crafting
```

### Training Models
To train a model, first update the corresponding configuration file in the `configs` folder with the path to the created dataset (should be fine if you ran the command listed above) and the desired parameters. Then run the following command:
```
python scripts/train.py --config configs/path/to/config --save-path path/to/output
```

### Monitoring Jobs
Jobs automatically output tensorboard logs to the specified location. You can view them using tensorboard.


## TODOS

Update the create_crafting_dataset
Update the configs
Test everything.