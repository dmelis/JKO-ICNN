################################################################################
########################   Code to run JKO-ICNN   ##############################
################################################################################

################################################################################
# The code is released exclusively for review purposes with the following terms:
# PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED USE, COPYING, OR DISTRIBUTION OF THE CODE,
# VIA ANY MEDIUM, IS STRICTLY PROHIBITED. BY ACCESSING THE CODE, THE REVIEWERS AGREE TO
# DELETE THEM FROM ALL MEDIA AFTER THE REVIEW PERIOD IS OVER.

# This repository is intended for reviewers to have access to the code that generated results
# and plots in the paper submission.
################################################################################


# Note: This codebase relies on some functions adapted from the CP-Flows paper,
# included here for convenience.
# Original repo: https://github.com/CW-Huang/CP-Flow/blob/main/lib/flows/cpflows.py

# For the molecular discovery experiment, we also use a VAE model from the MOSES repository,
# which is included here as well in the moses directory.
# Original repo: https://github.com/molecularsets/moses

### Install

# From top-level directory, runn the following to create and activate virtual env:
python3 -m venv ./jkoicnn_env
source jkoicnn_env/bin/activate

# From top-level directory, run the following (ideally from a virtual env):
pip install -r requirements.txt

# Run the following to enable the virtual env as notebook kernel:
python -m ipykernel install --user --name=jkoicnn_env
# For the notebooks below, ensure this virtual env is being used as the kernel:
Kernel -> Change kernel -> jkoicnn_env

# ffmpeg is needed to generate flow videos. If not already installed, it can be
# installed in mac via:
brew install ffmpeg


### Instructions to run PDE experiments

# Run:
jupyter notebook notebooks/Experiments_PDE.ipynb
# (make sure that the virtual env from above "jkoicnn_env" is being used for the notebook kernel)

### Instructions to run molecule experiments.

# All of the relevant files (saved model weights and sample point cloud) are already saved
# in the molecule_pde_files directory.
# Run
jupyter notebook notebooks/Molecule_Distribution_PDE.ipynb
# (make sure that the virtual env from above "jkoicnn_env" is being used for the notebook kernel)
