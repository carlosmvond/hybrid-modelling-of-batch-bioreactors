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

## Python library

The `hybrid_bioreactors` package provides a lightweight API to train
user-defined hybrid models with pandas dataframes. The dataframe is
expected to follow this structure:

1. The **first column** contains the experiment identifier.
2. The remaining columns contain time and measurement values.

Users provide the column names via `DataSpec` and define which
experiments to use for training, validation, and testing via
`ExperimentSplit`.

```python
import pandas as pd
from hybrid_bioreactors import DataSpec, ExperimentSplit, HybridTrainer, TrainingConfig

class MyHybridModel:
    def fit(self, data, spec, config):
        print("Training with", config.algorithm)

    def predict(self, data, spec):
        return data[[spec.experiment_col, spec.time_col]]

spec = DataSpec(
    experiment_col="experiment",
    time_col="time",
    measurement_cols=["biomass", "substrate"],
)

split = ExperimentSplit(train=[1, 2, 3], validation=[4], test=[5])
config = TrainingConfig(algorithm="minibatch", max_step=0.05, batch_size=32)

trainer = HybridTrainer(MyHybridModel(), spec, config)
outcome = trainer.train(df, split)
```

## Requirements

The notebooks are written in Python and rely mainly on:

JAX

Optax

NumPy

Matplotlib

The library requires:

pandas

They were developed and tested in Google Colab but can be run locally
with a compatible JAX installation.

## Purpose

These examples accompany a research paper on hybrid modeling with
limited batch data and are intended to illustrate model construction,
training strategies, and interpretation of learned dynamics.
