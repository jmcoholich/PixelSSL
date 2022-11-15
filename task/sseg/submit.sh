#!/bin/bash
#SBATCH --job-name=suponly_val
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=short
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/suponly_val.out
#SBATCH --error=slurm_logs/suponly_val.err
#SBATCH --constraint=2080_ti

set -x

# source /nethome/jcoholich3/.bashrc
# conda init bash
# conda deactivate
source /nethome/jcoholich3/miniconda3/etc/profile.d/conda.sh

conda activate PixelSSL


echo "Launching training"
# python -m script.deeplabv2_pascalvoc_1-16_sslmt
# python -m script.deeplabv2_pascalvoc_1-16_sslmt_large_batch
python -m script.deeplabv2_pascalvoc_1-16_suponly


