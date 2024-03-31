#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "1 False False 10 10 1 1"
#SBATCH -C A100|V100
module load cuda
source ./venv/bin/activate
python ./main.py 1 False False 10 10 1 1
