#!/bin/bash -l
#SBATCH --job-name=tensorbord-trial
#SBATCH --nodes=1
#SBATCH --account=env540
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output tensorboard-log-%J.out

module load gcc python openmpi py-tensorflow

# Activate virtual environment or conda environment
source ~/venvs/ipeo_venv/bin/activate # Replace with your environment setup

ipnport=$(shuf -i8000-9999 -n1)
tensorboard --logdir results/runs/test1/logs --port=${ipnport} --bind_all

# Run your Python script
python IPEO-Project/models/U-NET/train.py --config IPEO-Project/models/U-NET/config/train.yaml