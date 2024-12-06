#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=70g
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
python -m tasks run-multi-structure-pipeline --model_size='large' --problem_name='anchored_suspended_bridge_128x128_0.1' --num_trials=20