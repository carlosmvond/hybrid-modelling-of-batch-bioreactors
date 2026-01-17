"""Training utilities for user-defined hybrid models.

The library assumes a batch dataframe where the first column identifies
experimental runs, followed by time and measurement columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import pandas as pd


@dataclass(frozen=True)
class DataSpec:
    """Specification for mapping dataframe columns to model inputs."""

    experiment_col: str
    time_col: str
    measurement_cols: Sequence[str]

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe follows the expected format."""
        expected_cols = [self.experiment_col, self.time_col, *self.measurement_cols]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.columns[0] != self.experiment_col:
            raise ValueError(
                "The experiment column must be the first column in the dataframe."
            )


@dataclass(frozen=True)
class ExperimentSplit:
    """Defines experiment IDs assigned to train/validation/test splits."""

    train: Sequence[str | int]
    validation: Sequence[str | int] = field(default_factory=list)
    test: Sequence[str | int] = field(default_factory=list)

    def validate(self) -> None:
        """Ensure splits do not overlap."""
        train_set = set(self.train)
        validation_set = set(self.validation)
        test_set = set(self.test)
        overlap = (train_set & validation_set) | (train_set & test_set) | (
            validation_set & test_set
        )
        if overlap:
            raise ValueError(
                "Experiment IDs cannot appear in multiple splits. "
                f"Overlaps found: {sorted(overlap)}"
            )

    def all_experiments(self) -> List[str | int]:
        """Return all experiment IDs in the split."""
        return [*self.train, *self.validation, *self.test]


@dataclass
class TrainingConfig:
    """Configuration for the hybrid training algorithm."""

    algorithm: str = "global"
    max_step: float = 0.1
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    batch_size: int = 16
    shuffle: bool = True
    random_state: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration values."""
        if self.algorithm not in {"minibatch", "global"}:
            raise ValueError("Algorithm must be either 'minibatch' or 'global'.")
        if self.max_step <= 0:
            raise ValueError("max_step must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")


class HybridModel(Protocol):
    """Protocol for user-defined hybrid models."""

    def fit(self, data: pd.DataFrame, spec: DataSpec, config: TrainingConfig) -> None:
        """Fit the model to training data."""

    def predict(self, data: pd.DataFrame, spec: DataSpec) -> pd.DataFrame:
        """Generate predictions for the provided data."""


@dataclass
class TrainingOutcome:
    """Outcome of training, including prepared datasets and predictions."""

    train_data: pd.DataFrame
    validation_data: Optional[pd.DataFrame]
    test_data: Optional[pd.DataFrame]
    validation_predictions: Optional[pd.DataFrame]
    test_predictions: Optional[pd.DataFrame]


class HybridTrainer:
    """Coordinator for data preparation and training."""

    def __init__(
        self, model: HybridModel, spec: DataSpec, config: Optional[TrainingConfig] = None
    ) -> None:
        self.model = model
        self.spec = spec
        self.config = config or TrainingConfig()

    def prepare_split(self, df: pd.DataFrame, split: ExperimentSplit) -> Dict[str, pd.DataFrame]:
        """Filter the dataframe into train/validation/test subsets."""
        self.spec.validate(df)
        split.validate()
        self.config.validate()

        experiment_values = set(df[self.spec.experiment_col].unique())
        missing_experiments = [
            exp_id for exp_id in split.all_experiments() if exp_id not in experiment_values
        ]
        if missing_experiments:
            raise ValueError(f"Unknown experiment IDs: {missing_experiments}")

        datasets = {
            "train": self._filter_experiments(df, split.train),
            "validation": self._filter_experiments(df, split.validation),
            "test": self._filter_experiments(df, split.test),
        }
        return datasets

    def train(self, df: pd.DataFrame, split: ExperimentSplit) -> TrainingOutcome:
        """Train the model and optionally generate validation/test predictions."""
        datasets = self.prepare_split(df, split)
        train_data = datasets["train"]
        validation_data = datasets["validation"]
        test_data = datasets["test"]

        self.model.fit(train_data, self.spec, self.config)

        validation_predictions = (
            self._predict_if_available(validation_data)
            if not validation_data.empty
            else None
        )
        test_predictions = (
            self._predict_if_available(test_data) if not test_data.empty else None
        )

        return TrainingOutcome(
            train_data=train_data,
            validation_data=validation_data if not validation_data.empty else None,
            test_data=test_data if not test_data.empty else None,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
        )

    def _filter_experiments(
        self, df: pd.DataFrame, experiments: Iterable[str | int]
    ) -> pd.DataFrame:
        if not experiments:
            return df.iloc[0:0].copy()
        mask = df[self.spec.experiment_col].isin(list(experiments))
        return df.loc[mask].copy()

    def _predict_if_available(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        if hasattr(self.model, "predict"):
            return self.model.predict(data, self.spec)
        return None
