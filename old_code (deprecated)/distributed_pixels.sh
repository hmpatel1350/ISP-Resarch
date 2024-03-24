#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -J "distributed 1 False False 10 10 mse no-sigmoid"
#SBATCH -C A100|V100
module load cuda
source ./venv/bin/activate
python ./distributed_pixels.py 1 False False 10 10
