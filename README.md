# Hybrid modeling of batch bioreactors

This repository contains the code and supplementary data accompanying the paper
*Hybrid modeling framework for bioprocesses with minimal qualitative knowledge and
scarce data*. The framework combines ordinary differential equations with a feedforward
neural network, and builds the model structure from qualitative biological principles
(non-negativity, absence of spontaneous generation and biomass-mediated interactions)
rather than from mass balances. Models are trained from a few batch experiments using a
minibatch strategy in which each gradient update is computed from a single batch run.

## Contents

### Notebooks

**`sweep_case_study_1_Ecoli_2026.ipynb`** — Case Study 1. Hybrid modeling of
*Escherichia coli* growth on glucose with overflow metabolism (acetate production),
using synthetic batch data generated with the model of Mauri et al. (2020). The notebook
contains the hyperparameter sweep, the comparison between the minibatch and global
training strategies, the training of a single configuration, and the figures:
trajectories with Monte Carlo dropout uncertainty bands and the learned relationship
between the specific growth rate and the specific acetate production rate.

**`sweep_case_study_2_yeast_2026.ipynb`** — Case Study 2. Hybrid modeling of
astaxanthin production by *Xanthophyllomyces dendrorhous*, using the experimental batch
data of Liu and Wu (2008). The notebook trains the proposed model and, under identical
conditions, a hybrid model derived from mass balances, and compares their predictive
performance and uncertainty. It also produces the trajectories with uncertainty bands
and the interpretability figures.

### Supplementary tables

Complete results of the sensitivity analyses, of which only representative cases are
reported in the paper. Each row corresponds to one training run.

- **`TableS1.xlsx`** — Comparison of the minibatch and global strategies for Case
  Study 1 (32 runs).
- **`TableS2.xlsx`** — Complete sweep of Case Study 1: 72 configurations with three
  random initializations each (216 runs).
- **`TableS3.xlsx`** — Complete sweep of Case Study 2, with one sheet per model (96
  runs each).

## Requirements

The notebooks are written in Python and rely mainly on:

- JAX
- Diffrax
- Optax
- NumPy
- Matplotlib
- pandas and openpyxl (for the Excel output)

They were developed and tested in Google Colab, but can be run locally with a
compatible JAX installation. Missing dependencies are installed automatically on first
run.

## Data

The synthetic data of Case Study 1 are generated within the notebook. The experimental
data of Case Study 2 were taken from:

- Liu, Y.-S. & Wu, J.-Y. (2008). Modeling of *Xanthophyllomyces dendrorhous* growth on
  glucose and overflow metabolism in batch and fed-batch cultures for astaxanthin
  production. *Biotechnology and Bioengineering*, 101(5), 996–1004.

The synthetic data follow the model of:

- Mauri, M., Gouzé, J.-L., de Jong, H. & Cinquemani, E. (2020). Enhanced production of
  heterologous proteins by a synthetic microbial community: Conditions and trade-offs.
  *PLOS Computational Biology*, 16(4), e1007795.
