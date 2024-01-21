#!/bin/bash
# 
# Installer for package
# 
# Run: ./install.sh
# 


echo 'Creating seis2rock environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate seis2rock
conda env list
echo 'Created and activated environment:' $(which python)

# check pylops works as expected
echo 'Checking pylops version and running a command...'
python -c 'import numpy as np; import pylops; print(pylops.__version__); pylops.Identity(10) * np.ones(10)'

echo 'Done!'

