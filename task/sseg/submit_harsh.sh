#!/bin/bash
#SBATCH --job-name=suponly_city
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=short
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/suponly_city.out
#SBATCH --error=slurm_logs/suponly_city.err
#SBATCH --constraint=2080_ti

set -x

# source /nethome/jcoholich3/.bashrc
# conda init bash
# conda deactivate
source /nethome/hmaheshwari7/.bashrc
conda deactivate
conda activate PixelSSLenv


echo "Launching training"
# python -m script.deeplabv2_pascalvoc_1-16_sslmt
# python -m script.deeplabv2_pascalvoc_1-16_sslmt_large_batch
python -m script.deeplabv2_cityscapes_1-16_suponly


