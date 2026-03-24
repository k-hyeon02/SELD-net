#!/bin/bash
#SBATCH --job-name=seld-net
#SBATCH --partition=sichpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/seld_%j.out
#SBATCH --error=logs/seld_%j.err

# task-id 인자 받기 (기본값: 1)
# 사용법: sbatch run_seld.sh [task-id]
# task-id: 1=default, 2=ansim, 3=resim, 4=cansim, 5=cresim, 6=real, 7=cansim_ov3, 8=ansim_seq64, 999=quicktest
TASK_ID=${1:-1}
JOB_ID=${SLURM_JOB_ID:-0}

echo "====================================="
echo "SLURM Job ID : $JOB_ID"
echo "Task ID      : $TASK_ID"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time   : $(date)"
echo "====================================="

# 로그 디렉토리 생성
mkdir -p logs

# conda 환경 활성화
source /opt/shared/anaconda3/etc/profile.d/conda.sh
conda activate dlearn

cd /home/s2021102349/seld-net-master

python seld_train.py $JOB_ID $TASK_ID

echo "====================================="
echo "End time : $(date)"
echo "====================================="
