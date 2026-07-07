#!/bin/bash
#SBATCH --job-name=BP_OSD_Decoder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mail-user=harsh.kapoor@research.iiit.ac.in
#SBATCH --mail-type=ALL


echo "START"
date
hostname

source /home2/harsh.kapoor/general/bin/activate

# Run the script using the environment's python
python Decoder.py

echo "END"
date