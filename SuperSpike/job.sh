#!/bin/bash
#SBATCH -p gpu_x2
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH --mem 20G
#SBATCH --job-name=superspike
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-user=divyansh.gupta@students.iiserpune.ac.in
#SBATCH --mail-type=ALL

echo "loading cuda"
module load cuda-11.0.2-gcc-10.2.0-3wlbq6u
echo "cuda loaded"
conda activate div
srun -v python3 -v main_single_neuron.py
