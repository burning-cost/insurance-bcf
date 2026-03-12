"""Tests for simulate module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_bcf.simulate import (
    SimulationParams,
    SimulatedDataset,
    simulate_renewal,
    simulate_continuous_outcome,
)


class TestSimulationParams:
    def test_defaults(self):
        p = SimulationParams()
        assert p.n_policies == 2000
        assert p.n_features == 10
        assert 0 < p.treatment_fraction < 1
        assert 0 < p.base_renewal_prob < 1

    def test_custom(self):
        p = SimulationParams(n_policies=100, n_features=7, random_seed=99)
        assert p.n_policies == 100
        assert p.n_features == 7
        assert p.random_seed == 99


class TestSimulateRenewal:
    def test_returns_dataset(self):
        data = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))
        assert isinstance(data, SimulatedDataset)

    def test_shape_matches_n_policies(self):
        n = 300
        data = simulate_renewal(SimulationParams(n_policies=n, random_seed=1))
        assert len(data.X) == n
        assert len(data.treatment) == n
        assert len(data.outcome) == n
        assert len(data.true_tau) == n
        assert len(data.true_propensity) == n

    def test_feature_columns(self):
        data = simulate_renewal(SimulationParams(n_policies=50, n_features=5, random_seed=2))
        assert "age_band" in data.X.columns
        assert "ncb_steps" in data.X.columns
        assert "vehicle_age" in data.X.columns
        assert "channel" in data.X.columns
        assert "policy_duration" in data.X.columns
        assert data.X.shape[1] == 5

    def test_extra_noise_features(self):
        data = simulate_renewal(SimulationParams(n_policies=50, n_features=8, random_seed=3))
        assert data.X.shape[1] == 8
        assert "noise_0" in data.X.columns
        assert "noise_2" in data.X.columns

    def test_binary_outcome(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=4))
        unique = set(data.outcome.unique())
        assert unique.issubset({0.0, 1.0})

    def test_binary_treatment(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=5))
        unique = set(data.treatment.unique())
        assert unique.issubset({0.0, 1.0})

    def test_treatment_fraction_approx(self):
        data = simulate_renewal(SimulationParams(
            n_policies=1000, treatment_fraction=0.4, random_seed=6
        ))
        frac = data.treatment.mean()
        # Within 10pp of target (assignment via propensity threshold)
        assert 0.3 <= frac <= 0.5

    def test_propensity_in_unit_interval(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=7))
        pi = data.true_propensity
        assert (pi > 0).all()
        assert (pi < 1).all()

    def test_tau_heterogeneous(self):
        """True tau should vary across policies."""
        data = simulate_renewal(SimulationParams(n_policies=500, random_seed=8))
        assert data.true_tau.std() > 0.001

    def test_young_pcw_most_sensitive(self):
        """Young (age_band=0) PCW (channel=1) should have most negative tau."""
        data = simulate_renewal(SimulationParams(n_policies=1000, random_seed=9))
        young_pcw = data.true_tau[(data.X["age_band"] == 0) & (data.X["channel"] == 1)]
        old_direct = data.true_tau[(data.X["age_band"] == 5) & (data.X["channel"] == 0)]
        if len(young_pcw) > 5 and len(old_direct) > 5:
            assert young_pcw.mean() < old_direct.mean()

    def test_reproducible_with_seed(self):
        p = SimulationParams(n_policies=100, random_seed=42)
        d1 = simulate_renewal(p)
        d2 = simulate_renewal(p)
        pd.testing.assert_series_equal(d1.outcome, d2.outcome)

    def test_different_seeds_differ(self):
        d1 = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))
        d2 = simulate_renewal(SimulationParams(n_policies=100, random_seed=1))
        assert not d1.outcome.equals(d2.outcome)

    def test_no_nan_in_features(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=10))
        assert not data.X.isnull().any().any()

    def test_kwargs_override(self):
        data = simulate_renewal(n_policies=50, random_seed=11)
        assert len(data.X) == 50


class TestSimulateContinuous:
    def test_returns_dataset(self):
        data = simulate_continuous_outcome(n_policies=100, random_seed=0)
        assert isinstance(data, SimulatedDataset)

    def test_outcome_continuous(self):
        data = simulate_continuous_outcome(n_policies=200, random_seed=1)
        # Loss ratio — should not be all 0/1
        unique = data.outcome.unique()
        assert len(unique) > 2

    def test_loss_ratio_positive(self):
        data = simulate_continuous_outcome(n_policies=200, random_seed=2)
        assert (data.outcome > 0).all()

    def test_shape(self):
        data = simulate_continuous_outcome(n_policies=150, random_seed=3)
        assert len(data.X) == 150
        assert len(data.outcome) == 150
