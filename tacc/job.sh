#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o coulumb_teset.%j.out
#SBATCH -J coulumb_test
#SBATCH -p normal
#SBATCH -A DMS23021
#SBATCH -t 5:00:00

#SBATCH --mail-user=rodrigogonzalez@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

conda init
conda activate petrov_env
python3 ../parallel/parallel.py
