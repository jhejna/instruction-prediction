#!/bin/bash

#SBATCH --partition=napoli-gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=/cvgl2/u/jhejna/slurm_logs/%A.out
#SBATCH --error=/cvgl2/u/jhejna/slurm_logs/%A.err
#SBATCH --job-name="fomaml_ac_adapt"

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source /cvgl2/u/jhejna/condas/napoli/bin/activate
conda activate language_prediction

cd /cvgl2/u/jhejna/language-prediction

python scripts/train.py --config configs/lang_adapt/crafting_fomaml.yaml --save-path /cvgl2/u/jhejna/output/langauge_prediction/meta_12_1_adapt/fomaml_ac_adapt
