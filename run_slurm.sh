#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint=a6000
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=slurm/lora-rm-%j.out
#SBATCH --err=slurm/lora-rm-%j.err
#SBATCH --job-name=lora-rm-no-personalization
#SBATCH --mail-type=all
#SBATCH --mail-user=pw7nc@virginia.edu

source ~/.bashrc
source /p/finetunellm/anaconda3/bin/activate /p/finetunellm/anaconda3

bash run_reward_modeling.sh
