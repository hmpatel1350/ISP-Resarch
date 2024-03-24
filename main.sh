#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "2 True True 1 1 2 2"
#SBATCH -C A100|V100
module load cuda
source ./venv/bin/activate
python ./main.py 2 True True 1 1 2 2
