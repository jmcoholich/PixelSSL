#!/bin/bash
#SBATCH --job-name=suponly_224im_sliding_val_1_stride
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-gpu 7
#SBATCH --partition=short
#SBATCH --exclude=nestor
#SBATCH --output=slurm_logs/suponly_224im_sliding_val_1_stride.out
#SBATCH --error=slurm_logs/suponly_224im_sliding_val_1_stride.err
#SBATCH --constraint=rtx_6000

set -x

# source /nethome/jcoholich3/.bashrc
# conda init bash
# conda deactivate
source /nethome/jcoholich3/miniconda3/etc/profile.d/conda.sh

conda activate PixelSSL


echo "Launching training"
# python -m script.deeplabv2_pascalvoc_1-16_sslmt
# python -m script.deeplabv2_pascalvoc_1-16_sslmt_large_batch
# python -m script.deeplabv2_pascalvoc_1-16_suponly
# python -m script.deeplabv2_pascalvoc_1-16_suponly_160im
python -m script.deeplabv2_pascalvoc_1-16_suponly_224im


