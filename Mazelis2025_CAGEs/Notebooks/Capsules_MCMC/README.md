
# Capsules_MCMC - readme

This directory contains the cleaned notebooks and scripts used
for capsule-based analysis and 1D MCMC modeling in the Mazelis et al. (2025) study.


## CONTENTS

- *script1_Capsule_processing.ipynb*
  Preprocessing and analysis of single-cell data, supporting Fig. 4 of the paper.

- *script2_MCMC.ipynb*
  Training and evaluation of a 1D MCMC surrogate model based on simulated data, supporting Fig. 5 of the paper.

- *script1_Capsule_processing.py, script2_MCMC.py*
  Script equivalents of the notebooks (used for linting and non-interactive runs).

- *gene_lists/*
  Gene lists and filtering rules used for plotting and NMF analysis.

- *Table_LibraryBatches.xlsx*
  Sample and batch metadata required for preprocessing.

- *environment_script1.yml*
  Conda environment for capsule preprocessing.

- *environment_script2_pymc_arm.yml*
  Conda environment for PyMC-based MCMC analysis (Apple Silicon).

------------------------------------------------------------

## REQUIRED INPUT DATA (NOT INCLUDED)

The notebooks expect access to preprocessed AnnData (.h5ad) files containing
single-cell expression data. These files are not included in this repository
due to size constraints.

Please download the required data from the associated Zenodo record
(see the manuscript or the repository root README) and place them in a directory
named:

  ./Not_normalized/

relative to this folder.

------------------------------------------------------------

## ENVIRONMENT SETUP

Capsule preprocessing:

  conda env create -f environment_script1.yml
  conda activate script1

MCMC analysis (Apple Silicon):

  conda env create -f environment_script2_pymc_arm.yml
  conda activate pymc_arm

------------------------------------------------------------

## NOTES

- All paths in the notebooks are relative and assume execution from this directory.
- Large intermediate files, caches, and derived results are intentionally excluded.
- The notebooks were cleaned using static analysis tools (ruff, vulture) to improve
  code clarity and reproducibility.

------------------------------------------------------------

CITATION

If you use these materials, please cite:

Mazelis et al., 2025. Multi-step genomics on single cells and live cultures in sub-nanoliter capsules.
