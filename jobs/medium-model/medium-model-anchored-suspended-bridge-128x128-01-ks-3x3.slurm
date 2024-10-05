#!/bin/bash -l
#SBATCH --time=16:00:00
#SBATCH --ntasks=8
#SBATCH --mem=65g
#SBATCH --tmp=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
/home/jusun/dever120/NCVX-Neural-Structural-Optimization
export PATH=/home/jusun/dever120/miniconda3/envs/rydevera/bin:$PATH
python -m tasks run-multi-structure-pipeline-v2 --model_size='medium' --problem_name='anchored_suspended_bridge_128x128_0.1' --kernel_size="3,3"