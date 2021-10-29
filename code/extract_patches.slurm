#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=4-09:00:00
#SBATCH --mail-user=$axa1399@case.edu
#SBATCH --job-name="extractPatches"
#SBATCH --output="log.pipeline.extractPatches"


#=========loading libraries==========#
module swap intel gcc
module load python/3.7.0
module load openslide
#pip3.6 install --upgrade setuptools pip --user
#pip3.6 install openslide-python --user
#pip3.6 install opencv-python --user


#===========patch extraction============#
time python3 extract_patches.py