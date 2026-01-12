# hybrid-modelling-of-batch-bioreactors

This repository contains two Jupyter notebooks illustrating a hybrid
ODE–neural network modeling framework for batch bioprocess data.

Contents

Application_1_Ecoli_2026.ipynb
Hybrid modeling of Escherichia coli growth on glucose using synthetic
batch data, including overflow metabolism (acetate production).
The notebook compares minibatch and global training strategies and
analyzes model interpretability.

Application_2_yeast_2026.ipynb
Hybrid modeling of astaxanthin production by Xanthophyllomyces
dendrorhous using experimental batch data.
The notebook demonstrates model construction with minimal prior
knowledge and evaluates predictive performance and interpretability.

Requirements

The notebooks are written in Python and rely mainly on:

JAX

Optax

NumPy

Matplotlib

They were developed and tested in Google Colab but can be run locally
with a compatible JAX installation.

Purpose

These examples accompany a research paper on hybrid modeling with
limited batch data and are intended to illustrate model construction,
training strategies, and interpretation of learned dynamics.
