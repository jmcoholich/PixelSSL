#!/bin/bash
#SBATCH --job-name=pixel_mt
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=long
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/sslmt.out
#SBATCH --error=slurm_logs/sslmt.err

set -x

# source /nethome/jcoholich3/.bashrc
# conda init bash
# conda deactivate
source /nethome/jcoholich3/miniconda3/etc/profile.d/conda.sh

conda activate PixelSSL


echo "Launching training"
python -m script.deeplabv2_pascalvoc_1-16_sslmt


