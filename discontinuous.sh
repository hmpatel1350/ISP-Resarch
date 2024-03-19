#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "discontinuous 2 True True 10 10"
#SBATCH -C A100|V100
module load cuda
source ./venv/bin/activate
python ./discontinuous.py 2 True True 10 10
