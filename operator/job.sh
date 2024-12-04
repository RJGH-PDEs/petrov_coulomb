#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o operator.%j.out
#SBATCH -J operator_57_dense
#SBATCH -p normal
#SBATCH -A DMS23021
#SBATCH -t 9:30:00

#SBATCH --mail-user=rodrigogonzalez@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

conda init
conda activate petrov_env
python3 parallel.py
