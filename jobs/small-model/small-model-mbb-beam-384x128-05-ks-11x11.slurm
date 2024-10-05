#!/bin/bash -l
#SBATCH --time=16:00:00
#SBATCH --ntasks=8
#SBATCH --mem=65g
#SBATCH --tmp=40g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
/home/jusun/dever120/NCVX-Neural-Structural-Optimization
export PATH=/home/jusun/dever120/miniconda3/envs/rydevera/bin:$PATH
python -m tasks run-multi-structure-pipeline-v2 --model_size='small' --problem_name='mbb_beam_384x128_0.5' --kernel_size="11,11"