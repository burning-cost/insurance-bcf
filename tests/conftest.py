"""
Pytest fixtures shared across all test modules.

All tests use synthetic data from simulate_renewal(). The mock BCFModel
is used unless stochtree is genuinely available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_bcf.simulate import simulate_renewal, SimulationParams


@pytest.fixture(scope="session")
def small_dataset():
    """200-policy binary outcome dataset for fast tests."""
    return simulate_renewal(SimulationParams(n_policies=200, random_seed=0))


@pytest.fixture(scope="session")
def medium_dataset():
    """500-policy binary outcome dataset."""
    return simulate_renewal(SimulationParams(n_policies=500, random_seed=1))


@pytest.fixture(scope="session")
def fitted_model(small_dataset):
    """Pre-fitted BayesianCausalForest on the small dataset."""
    from insurance_bcf import BayesianCausalForest
    model = BayesianCausalForest(
        outcome="binary",
        num_mcmc=50,
        num_gfr=5,
        random_seed=42,
    )
    model.fit(
        small_dataset.X,
        small_dataset.treatment,
        small_dataset.outcome,
    )
    return model


@pytest.fixture(scope="session")
def fitted_estimator(fitted_model, small_dataset):
    """Pre-fitted ElasticityEstimator."""
    from insurance_bcf import ElasticityEstimator
    return ElasticityEstimator(fitted_model)
