#!/bin/bash
#SBATCH --job-name=twoteacher
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=short
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/twoteacher.out
#SBATCH --error=slurm_logs/twoteacher.err
#SBATCH --constraint=rtx_6000

set -x

source ~/.bashrc
conda activate PixelSSLenv
cd /srv/flash1/hmaheshwari7/semi-supervised/segmentation/PixelSSL/task/sseg

echo "Launching training"
python -m script.deeplabv2_pascalvoc_1-16_ssltwoteacher
