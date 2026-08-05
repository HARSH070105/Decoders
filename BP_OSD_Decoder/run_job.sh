#!/bin/bash
#SBATCH --job-name=BP_OSD_Decoder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=50:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0
#SBATCH --mail-user=harsh.kapoor@research.iiit.ac.in
#SBATCH --mail-type=ALL


echo "START"
date
hostname

source /home2/harsh.kapoor/general/bin/activate

# Run the script using the environment's python
python simulate.py --code_name ghp_882_24 --error_rates 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 --trials 500000
python simulate.py --code_name gb_254_28 --error_rates 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 --trials 500000
python simulate.py --code_name gb_126_28 --error_rates 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 --trials 500000

echo "END"
date