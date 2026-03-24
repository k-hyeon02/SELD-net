#!/bin/bash
#SBATCH --job-name=seld-feat
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --output=logs/feat_%j.out
#SBATCH --error=logs/feat_%j.err

echo "====================================="
echo "SLURM Job ID : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "====================================="

mkdir -p logs

source /opt/shared/anaconda3/etc/profile.d/conda.sh
conda activate dlearn

cd /home/s2021102349/seld-net-master

python batch_feature_extraction.py

echo "====================================="
echo "End time : $(date)"
echo "====================================="
