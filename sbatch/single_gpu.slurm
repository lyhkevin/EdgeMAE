#!/bin/bash
#BATCH -J My%jtestjobs
#SBATCH -p bme_gpu
#SBATCH -N 1#
#SBATCH -n 8
#SBATCH --gres=gpu:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -t 5-00:00:00

source ~/anaconda3/etc/profile.d/conda.sh
conda activate geometric
cd ../
python3 dp.py
