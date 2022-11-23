#!/bin/bash
#SBATCH --job-name=deeplabv2_pascalvoc_1-16_sslmtsliding
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=short
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/deeplabv2_pascalvoc_1-16_sslmtsliding.out
#SBATCH --error=slurm_logs/deeplabv2_pascalvoc_1-16_sslmtsliding.err
#SBATCH --constraint=rtx_6000

set -x

source ~/.bashrc
conda activate PixelSSL2
cd /nethome/skareer6/flash/Projects/SlidingTeacher/PixelSSL/task/sseg

echo "Launching training"
python -m script.deeplabv2_pascalvoc_1-16_sslmtsliding


