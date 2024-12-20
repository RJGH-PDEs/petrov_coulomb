#!/bin/bash

#SBATCH -N 1
#SBATCH -n 128
#SBATCH -o coulumb_1069.%j.out
#SBATCH -J coulumb_1069
#SBATCH -p normal
#SBATCH -A DMS23021
#SBATCH -t 07:00:00

#SBATCH --mail-user=rodrigogonzalez@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

conda init
python3 ../parallel/parallel.py
