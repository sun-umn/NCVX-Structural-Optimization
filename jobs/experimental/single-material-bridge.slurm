#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=75g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /users/5/dever120//NCVX-Neural-Structural-Optimization
export PATH=/users/5/dever120/.conda/envs/top-opt/bin:$PATH
echo $PATH
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate top-opt
python -m tasks test-single-material-bridge-v2