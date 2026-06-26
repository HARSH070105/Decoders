#!/bin/bash
#SBATCH --job-name=qecc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:0

echo "START"
date
hostname

workon general
uv run decoder.py

both echo "END" date



