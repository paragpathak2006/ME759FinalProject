#!/bin/bash
#SBATCH -p slurm_shortgpu
#SBATCH --job-name=p1
#SBATCH -N 1 -n 2 --gres=gpu:1
#SBATCH -o job-p1.o
#SBATCH module load cuda/6.0.26;

./problem1