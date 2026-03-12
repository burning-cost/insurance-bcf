"""
Pytest fixtures shared across all test modules.

All tests use synthetic data from simulate_renewal(). The mock BCFModel
is used unless stochtree is genuinely available.

The fitted_model fixture passes external propensity scores from the simulation
ground truth. This avoids positivity violations on small synthetic datasets
(where the logistic propensity estimator produces extreme scores) and mirrors
the recommended practice for insurance applications: pass a calibrated
external propensity rather than relying on the BCF internal estimate.
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
    """
    Pre-fitted BayesianCausalForest on the small dataset.

    Uses external (ground-truth) propensity scores to avoid positivity
    violations on the 200-policy synthetic dataset. On small datasets,
    logistic propensity estimation produces extreme scores for some
    policies; passing the simulation's true propensity (clipped to
    [0.05, 0.95]) avoids this without inflating the threshold.
    """
    from insurance_bcf import BayesianCausalForest
    model = BayesianCausalForest(
        outcome="binary",
        num_mcmc=50,
        num_gfr=5,
        random_seed=42,
    )
    # Use ground-truth propensity from simulation (already clipped to [0.05, 0.95])
    pi = small_dataset.true_propensity.to_numpy()
    model.fit(
        small_dataset.X,
        small_dataset.treatment,
        small_dataset.outcome,
        propensity=pi,
    )
    return model


@pytest.fixture(scope="session")
def fitted_estimator(fitted_model, small_dataset):
    """Pre-fitted ElasticityEstimator."""
    from insurance_bcf import ElasticityEstimator
    return ElasticityEstimator(fitted_model)
