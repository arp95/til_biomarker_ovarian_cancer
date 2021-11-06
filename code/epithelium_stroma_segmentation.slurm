#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60G
#SBATCH --time=09-00:00:00
#SBATCH --mail-user=$USER@case.edu
#SBATCH --job-name="epitheliumStroma"
#SBATCH --output="log.pipeline.epitheliumStroma"


#=========loading libraries==========#
module swap intel gcc
module load python/3.7.0
module load openslide
#pip3.6 install --upgrade setuptools pip --user
#pip3.6 install open slide-python --user
#pip3.6 install opencv-python --user


#===========patch extraction============#
time python3 /mnt/rstor/CSE_BME_AXM788/home/axa1399/til_biomarker_ovarian_cancer/code/epithelium_stroma_segmentation.py