#!/bin/bash
#SBATCH -J corrdiff
#SBATCH -p gpu1
#SBATCH -o %j.o
#SBATCH -e %j.e
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres gpu:1
#SBATCH --mem 300GB
#SBATCH -t 7-24:00

source activate corrdiff

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.

# ------------------ Sart model training ------------------
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

#-------------------  Inference  ------------------------
#python inference.py --config_json_path config.json --epoch_idx 15
# python test.py
