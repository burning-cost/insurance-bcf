"""Tests for ElasticityEstimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_bcf import BayesianCausalForest, ElasticityEstimator, NotFittedError
from insurance_bcf.simulate import SimulationParams, simulate_renewal


@pytest.fixture(scope="module")
def est_and_data():
    data = simulate_renewal(SimulationParams(n_policies=200, random_seed=10))
    model = BayesianCausalForest(num_mcmc=30, num_gfr=3, random_seed=0)
    model.fit(data.X, data.treatment, data.outcome)
    est = ElasticityEstimator(model)
    return est, data


class TestElasticityEstimatorConstructor:
    def test_accepts_fitted_model(self, fitted_model):
        est = ElasticityEstimator(fitted_model)
        assert est.model is fitted_model

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="BayesianCausalForest"):
            ElasticityEstimator("not a model")

    def test_default_credible_level(self, fitted_model):
        est = ElasticityEstimator(fitted_model)
        assert est.credible_level == 0.95

    def test_custom_credible_level(self, fitted_model):
        est = ElasticityEstimator(fitted_model, credible_level=0.80)
        assert est.credible_level == 0.80

    def test_unfitted_model_raises_on_use(self):
        model = BayesianCausalForest()
        est = ElasticityEstimator(model)
        data = simulate_renewal(SimulationParams(n_policies=50, random_seed=0))
        with pytest.raises(NotFittedError):
            est.segment_effects(data.X, ["age_band"])

    def test_repr(self, fitted_model):
        est = ElasticityEstimator(fitted_model)
        assert "ElasticityEstimator" in repr(est)


class TestSegmentEffects:
    def test_returns_dataframe(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["age_band"])
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["age_band"])
        for col in ["age_band", "effect_mean", "effect_lower", "effect_upper", "effect_std", "n_policies"]:
            assert col in result.columns

    def test_single_segment_col(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["channel"])
        # channel has 3 values (0,1,2) — expect up to 3 rows
        assert len(result) <= 3
        assert len(result) >= 1

    def test_multi_segment_col(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["age_band", "channel"])
        # 6 age_bands * 3 channels = up to 18 combos
        assert 1 <= len(result) <= 18

    def test_ci_bounds_ordered(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["age_band"])
        assert (result["effect_lower"] <= result["effect_mean"]).all()
        assert (result["effect_mean"] <= result["effect_upper"]).all()

    def test_n_policies_sum_le_total(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["channel"], min_policies=1)
        # n_policies per segment sums to total (no overlap)
        assert result["n_policies"].sum() <= len(data.X)

    def test_missing_col_raises(self, est_and_data):
        est, data = est_and_data
        with pytest.raises(ValueError, match="not in X"):
            est.segment_effects(data.X, ["nonexistent_col"])

    def test_min_policies_filters_small(self, est_and_data):
        est, data = est_and_data
        result_all = est.segment_effects(data.X, ["age_band"], min_policies=1)
        result_filtered = est.segment_effects(data.X, ["age_band"], min_policies=1000)
        assert len(result_filtered) <= len(result_all)

    def test_empty_when_all_filtered(self, est_and_data):
        est, data = est_and_data
        result = est.segment_effects(data.X, ["age_band"], min_policies=10000)
        assert result.empty

    def test_credible_level_override(self, est_and_data):
        est, data = est_and_data
        r90 = est.segment_effects(data.X, ["age_band"], credible_level=0.90)
        r99 = est.segment_effects(data.X, ["age_band"], credible_level=0.99)
        if not r90.empty and not r99.empty:
            w90 = (r90["effect_upper"] - r90["effect_lower"]).mean()
            w99 = (r99["effect_upper"] - r99["effect_lower"]).mean()
            assert w99 >= w90


class TestPartialDependence:
    def test_returns_dataframe(self, est_and_data):
        est, data = est_and_data
        result = est.partial_dependence(data.X, feature="age_band")
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, est_and_data):
        est, data = est_and_data
        result = est.partial_dependence(data.X, feature="age_band")
        for col in ["feature_value", "pdp_mean", "pdp_lower", "pdp_upper"]:
            assert col in result.columns

    def test_row_count_matches_unique_values(self, est_and_data):
        est, data = est_and_data
        n_unique = data.X["age_band"].nunique()
        result = est.partial_dependence(data.X, feature="age_band", grid_points=20)
        # age_band has 6 unique values < 20, so should return 6 rows
        assert len(result) == n_unique

    def test_continuous_feature_grid_points(self, est_and_data):
        est, data = est_and_data
        result = est.partial_dependence(
            data.X, feature="policy_duration", grid_points=10
        )
        assert len(result) == 10

    def test_ci_ordered(self, est_and_data):
        est, data = est_and_data
        result = est.partial_dependence(data.X, feature="ncb_steps")
        assert (result["pdp_lower"] <= result["pdp_mean"]).all()
        assert (result["pdp_mean"] <= result["pdp_upper"]).all()

    def test_missing_feature_raises(self, est_and_data):
        est, data = est_and_data
        with pytest.raises(ValueError, match="not in X.columns"):
            est.partial_dependence(data.X, feature="nonexistent")

    def test_subsampling(self, est_and_data):
        est, data = est_and_data
        result = est.partial_dependence(
            data.X, feature="age_band", n_sample_policies=50
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestOptimalRateAdjustment:
    def test_returns_dataframe(self, est_and_data):
        est, data = est_and_data
        premium = pd.Series(np.random.uniform(400, 1200, len(data.X)), index=data.X.index)
        result = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, est_and_data):
        est, data = est_and_data
        premium = pd.Series(np.full(len(data.X), 600.0), index=data.X.index)
        result = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium
        )
        for col in ["suggested_adjustment", "cate_mean", "cate_lower", "cate_upper",
                    "current_premium", "adjustment_confidence"]:
            assert col in result.columns

    def test_adjustments_within_max(self, est_and_data):
        est, data = est_and_data
        premium = pd.Series(np.full(len(data.X), 600.0), index=data.X.index)
        max_adj = 0.15
        result = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium, max_adjustment=max_adj
        )
        assert (result["suggested_adjustment"].abs() <= max_adj + 1e-6).all()

    def test_confidence_in_unit_interval(self, est_and_data):
        est, data = est_and_data
        premium = pd.Series(np.full(len(data.X), 600.0), index=data.X.index)
        result = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium
        )
        assert (result["adjustment_confidence"] >= 0).all()
        assert (result["adjustment_confidence"] <= 1).all()

    def test_index_preserved(self, est_and_data):
        est, data = est_and_data
        premium = pd.Series(np.full(len(data.X), 600.0), index=data.X.index)
        result = est.optimal_rate_adjustment(
            data.X, target_margin=0.05, current_premium=premium
        )
        pd.testing.assert_index_equal(result.index, data.X.index)


class TestPortfolioSummary:
    def test_returns_dataframe(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        assert isinstance(result, pd.DataFrame)

    def test_one_row(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        assert len(result) == 1

    def test_columns(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        for col in ["mean_cate", "median_cate", "p10_cate", "p90_cate",
                    "frac_negative_ci", "frac_positive_ci", "frac_significant", "n_policies"]:
            assert col in result.columns

    def test_fractions_in_unit_interval(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        assert 0 <= result["frac_negative_ci"].iloc[0] <= 1
        assert 0 <= result["frac_positive_ci"].iloc[0] <= 1
        assert result["frac_significant"].iloc[0] <= 1

    def test_n_policies_correct(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        assert result["n_policies"].iloc[0] == len(data.X)

    def test_quantile_ordering(self, est_and_data):
        est, data = est_and_data
        result = est.portfolio_summary(data.X)
        assert result["p10_cate"].iloc[0] <= result["median_cate"].iloc[0]
        assert result["median_cate"].iloc[0] <= result["p90_cate"].iloc[0]
