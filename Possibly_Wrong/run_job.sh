#!/bin/bash
#SBATCH --job-name=Tile_Code
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=37
#SBATCH --gres=gpu:0
#SBATCH --mail-user=harsh.kapoor@research.iiit.ac.in
#SBATCH --mail-type=ALL


echo "START"
date
hostname

source /home2/harsh.kapoor/general/bin/activate

python simulate_p_adjusted.py

echo "END"
date