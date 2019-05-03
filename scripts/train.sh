#!/bin/sh

#SBATCH --time=7-0:00:00

#. ~/tf-cpu-1.13/bin/activate
module load cuda/9.1
module load cuDNN/6.0
. /rhome/mizbicki/tf-gpu-1.4/bin/activate
python -u model/train.py $@
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBAT -p gpu
#SBAT --gres=gpu:p100:1
