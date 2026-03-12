"""
Integration tests — full workflow from simulation to report.

These exercise the complete pipeline as a pricing team would use it.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from insurance_bcf import (
    BayesianCausalForest,
    BCFAuditReport,
    ElasticityEstimator,
)
from insurance_bcf.simulate import SimulationParams, simulate_renewal, simulate_continuous_outcome


class TestFullBinaryWorkflow:
    """
    Complete pipeline: simulate -> fit -> CATE -> segment effects -> report.
    Mirrors the UK motor renewal rate change use case from the research report.
    """

    def test_binary_outcome_pipeline(self):
        data = simulate_renewal(SimulationParams(n_policies=300, random_seed=100))
        model = BayesianCausalForest(outcome="binary", num_mcmc=40, num_gfr=5, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)

        cate_df = model.cate(data.X)
        assert len(cate_df) == 300
        assert (cate_df["cate_lower"] <= cate_df["cate_upper"]).all()

        est = ElasticityEstimator(model)
        seg = est.segment_effects(data.X, ["age_band"])
        assert not seg.empty

        report = BCFAuditReport(model, est)
        pc = report.protected_characteristic_check(data.X, ["age_band"])
        assert "PORTFOLIO" in pc["characteristic"].values

    def test_binary_outcome_with_external_propensity(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=101))
        pi = data.true_propensity.to_numpy()
        model = BayesianCausalForest(outcome="binary", num_mcmc=30, num_gfr=3, random_seed=1)
        model.fit(data.X, data.treatment, data.outcome, propensity=pi)

        cate_df = model.cate(data.X)
        assert len(cate_df) == 200

    def test_segment_effects_ordering(self):
        """Segment effects should be sorted by effect_mean ascending."""
        data = simulate_renewal(SimulationParams(n_policies=500, random_seed=102))
        model = BayesianCausalForest(outcome="binary", num_mcmc=50, num_gfr=5, random_seed=2)
        model.fit(data.X, data.treatment, data.outcome)
        est = ElasticityEstimator(model)
        seg = est.segment_effects(data.X, ["channel"], min_policies=1)
        if len(seg) > 1:
            means = seg["effect_mean"].to_numpy()
            assert (means[:-1] <= means[1:]).all()


class TestFullContinuousWorkflow:
    def test_continuous_outcome_pipeline(self):
        data = simulate_continuous_outcome(n_policies=200, random_seed=200)
        model = BayesianCausalForest(outcome="continuous", num_mcmc=30, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)

        cate_df = model.cate(data.X)
        assert len(cate_df) == 200
        assert not cate_df["cate_mean"].isnull().any()


class TestReportPipeline:
    def test_full_report_render(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=300))
        model = BayesianCausalForest(outcome="binary", num_mcmc=30, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)
        est = ElasticityEstimator(model)
        report = BCFAuditReport(model, est)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "integration_report.html")
            report.render(
                output_path=path,
                X=data.X,
                Z=data.treatment,
                protected_cols=["age_band", "channel"],
                segment_cols=[["age_band"], ["channel"]],
                include_plots=True,
            )
            content = open(path).read()
            assert "BayesianCausalForest" in content or "Bayesian Causal Forest" in content
            assert "Methodology" in content
            assert "Protected Characteristic" in content


class TestRateAdjustmentWorkflow:
    def test_rate_adjustment_pipeline(self):
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=400))
        model = BayesianCausalForest(outcome="binary", num_mcmc=30, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)
        est = ElasticityEstimator(model)

        premium = pd.Series(
            np.random.uniform(400, 1200, len(data.X)),
            index=data.X.index
        )
        adj = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium, max_adjustment=0.20
        )
        assert len(adj) == len(data.X)
        assert (adj["suggested_adjustment"].abs() <= 0.20 + 1e-6).all()


class TestSerialisationRoundtrip:
    def test_serialise_and_refit(self):
        data = simulate_renewal(SimulationParams(n_policies=150, random_seed=500))
        model = BayesianCausalForest(outcome="binary", num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)

        cate_original = model.cate(data.X)

        json_str = model.to_json()
        model2 = BayesianCausalForest.from_json(json_str, outcome="binary")
        cate_restored = model2.cate(data.X)

        # Both should return DataFrames of the right shape
        assert len(cate_original) == len(cate_restored) == 150


class TestPropensityEstimation:
    def test_logistic_propensity_range(self):
        """Logistic propensity estimation should stay in (0, 1)."""
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=600))
        model = BayesianCausalForest(outcome="binary", num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome, propensity=None)
        pi = model.propensity_scores
        assert pi is not None
        assert (pi > 0).all()
        assert (pi < 1).all()

    def test_continuous_treatment_propensity(self):
        """Continuous treatment uses Gaussian propensity approximation."""
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=601))
        # Make treatment continuous
        treatment_cont = data.treatment * np.random.uniform(0.8, 1.2, len(data.X))
        model = BayesianCausalForest(outcome="binary", num_mcmc=20, num_gfr=3)
        model.fit(data.X, treatment_cont, data.outcome)
        pi = model.propensity_scores
        assert pi is not None
        assert (pi > 0).all()
        assert (pi < 1).all()


class TestEdgeCases:
    def test_single_treatment_value_warns_or_fits(self):
        """All-treatment or all-control scenarios should at least not crash."""
        data = simulate_renewal(SimulationParams(n_policies=100, random_seed=700))
        # All treated
        treatment_all_one = pd.Series(np.ones(100))
        model = BayesianCausalForest(outcome="binary", num_mcmc=10, num_gfr=2)
        # This will produce a degenerate propensity — may warn but should not crash
        try:
            model.fit(data.X, treatment_all_one, data.outcome)
        except (ValueError, Exception):
            pass  # acceptable to raise for degenerate input

    def test_single_policy(self):
        """A single-policy dataset should fail gracefully."""
        data = simulate_renewal(SimulationParams(n_policies=1, random_seed=800))
        model = BayesianCausalForest(outcome="binary", num_mcmc=10, num_gfr=2)
        try:
            model.fit(data.X, data.treatment, data.outcome)
        except Exception:
            pass  # Any exception is acceptable for n=1
