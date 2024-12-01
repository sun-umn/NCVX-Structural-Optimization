#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
cd /users/5/dever120/NCVX-Neural-Structural-Optimization
export PATH=/users/5/dever120/.conda/envs/general/bin:$PATH
echo $PATH
nvidia-smi
eval "$(conda shell.bash hook)"
conda activate general
python -m tasks run-multi-structure-pipeline-v2 --model_size='small' --problem_name='mbb_beam_96x32_0.5' --kernel_size="11,11"