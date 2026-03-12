"""Tests for BayesianCausalForest."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_bcf import (
    BayesianCausalForest,
    GIPPBreakWarning,
    NotFittedError,
    PositivityViolationError,
)
from insurance_bcf.simulate import SimulationParams, simulate_renewal, simulate_continuous_outcome


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    def test_defaults(self):
        model = BayesianCausalForest()
        assert model.outcome == "binary"
        assert model.treatment_trees == 50
        assert model.prognostic_trees == 250
        assert model.num_mcmc == 500
        assert model.num_gfr == 10
        assert model.num_burnin == 0
        assert model.num_chains == 1
        assert model.propensity_covariate == "prognostic"
        assert not model._is_fitted

    def test_custom_params(self):
        model = BayesianCausalForest(
            outcome="continuous",
            treatment_trees=30,
            prognostic_trees=100,
            num_mcmc=200,
            random_seed=7,
        )
        assert model.outcome == "continuous"
        assert model.treatment_trees == 30
        assert model.num_mcmc == 200
        assert model.random_seed == 7

    def test_invalid_outcome(self):
        with pytest.raises(ValueError, match="outcome must be"):
            BayesianCausalForest(outcome="probit")

    def test_propensity_none_warns(self):
        with pytest.warns(UserWarning, match="RIC correction"):
            BayesianCausalForest(propensity_covariate="none")

    def test_repr_unfitted(self):
        model = BayesianCausalForest()
        r = repr(model)
        assert "unfitted" in r
        assert "BayesianCausalForest" in r


# ------------------------------------------------------------------
# fit — input validation
# ------------------------------------------------------------------


class TestFitValidation:
    def setup_method(self):
        self.data = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))

    def test_x_not_dataframe(self):
        with pytest.raises(TypeError, match="pd.DataFrame"):
            BayesianCausalForest().fit(
                np.array(self.data.X),
                self.data.treatment,
                self.data.outcome,
            )

    def test_x_empty(self):
        with pytest.raises(ValueError, match="empty"):
            BayesianCausalForest().fit(
                pd.DataFrame(),
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
            )

    def test_x_nan(self):
        X_nan = self.data.X.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            BayesianCausalForest().fit(X_nan, self.data.treatment, self.data.outcome)

    def test_treatment_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            BayesianCausalForest().fit(
                self.data.X,
                self.data.treatment.iloc[:50],
                self.data.outcome,
            )

    def test_outcome_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            BayesianCausalForest().fit(
                self.data.X,
                self.data.treatment,
                self.data.outcome.iloc[:50],
            )

    def test_treatment_nan(self):
        Z_nan = self.data.treatment.copy()
        Z_nan.iloc[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            BayesianCausalForest().fit(self.data.X, Z_nan, self.data.outcome)

    def test_outcome_nan(self):
        y_nan = self.data.outcome.copy()
        y_nan.iloc[0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            BayesianCausalForest().fit(self.data.X, self.data.treatment, y_nan)

    def test_binary_outcome_with_non_binary_values(self):
        y_bad = self.data.outcome.copy().astype(float)
        y_bad.iloc[0] = 0.5
        with pytest.raises(ValueError, match="binary"):
            BayesianCausalForest(outcome="binary").fit(
                self.data.X, self.data.treatment, y_bad
            )

    def test_propensity_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            BayesianCausalForest().fit(
                self.data.X,
                self.data.treatment,
                self.data.outcome,
                propensity=np.array([0.5, 0.5]),  # wrong length
            )

    def test_propensity_out_of_bounds(self):
        bad_propensity = np.full(len(self.data.X), 0.5)
        bad_propensity[0] = 1.5  # > 1
        with pytest.raises(ValueError, match="strictly in"):
            BayesianCausalForest().fit(
                self.data.X,
                self.data.treatment,
                self.data.outcome,
                propensity=bad_propensity,
            )


# ------------------------------------------------------------------
# fit — successful paths
# ------------------------------------------------------------------


class TestFitSuccess:
    def setup_method(self):
        self.data = simulate_renewal(SimulationParams(n_policies=150, random_seed=42))

    def test_fit_returns_self(self):
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        result = model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert result is model

    def test_is_fitted_after_fit(self):
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert model._is_fitted

    def test_repr_fitted(self):
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert "fitted" in repr(model)

    def test_feature_names_stored(self):
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert model._feature_names == list(self.data.X.columns)

    def test_n_train_stored(self):
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert model._n_train == len(self.data.X)

    def test_fit_with_external_propensity(self):
        pi = self.data.true_propensity.to_numpy()
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome, propensity=pi)
        assert model._is_fitted
        np.testing.assert_array_equal(model.propensity_scores, pi)

    def test_fit_continuous_outcome(self):
        data = simulate_continuous_outcome(n_policies=100, random_seed=5)
        model = BayesianCausalForest(outcome="continuous", num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(data.X, data.treatment, data.outcome)
        assert model._is_fitted

    def test_fit_with_groups(self):
        groups = pd.Series(
            np.random.randint(1, 4, len(self.data.X)),
            name="insurer_id"
        )
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        # Groups trigger rfx_group_ids in BCF — mock supports it
        model.fit(self.data.X, self.data.treatment, self.data.outcome, groups=groups)
        assert model._is_fitted

    def test_fit_twice(self):
        """Fitting twice should re-fit (no stale state)."""
        model = BayesianCausalForest(num_mcmc=20, num_gfr=3, random_seed=0)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        model.fit(self.data.X, self.data.treatment, self.data.outcome)
        assert model._is_fitted


# ------------------------------------------------------------------
# positivity check
# ------------------------------------------------------------------


class TestPositivityCheck:
    def test_positivity_violation_raises(self):
        """Extreme propensity scores should raise PositivityViolationError."""
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=0))
        # Force extreme propensities
        bad_pi = np.full(200, 0.5)
        bad_pi[:20] = 0.01  # 10% below threshold
        with pytest.raises(PositivityViolationError):
            model = BayesianCausalForest(
                num_mcmc=10,
                positivity_max_fraction=0.05,
            )
            model.fit(data.X, data.treatment, data.outcome, propensity=bad_pi)

    def test_borderline_warns_not_raises(self):
        """A small number of violations should warn but not raise."""
        data = simulate_renewal(SimulationParams(n_policies=200, random_seed=1))
        pi = np.full(200, 0.5)
        pi[0] = 0.03  # 1 violation = 0.5% < default 5% threshold
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = BayesianCausalForest(num_mcmc=10, num_gfr=2)
            model.fit(data.X, data.treatment, data.outcome, propensity=pi)
        # Should complete without PositivityViolationError
        assert model._is_fitted


# ------------------------------------------------------------------
# GIPP break warning
# ------------------------------------------------------------------


class TestGIPPBreakWarning:
    def test_warns_when_spanning_gipp_date(self):
        data = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))
        X = data.X.copy()
        # Add date column spanning Jan 2022
        X["renewal_date"] = pd.date_range("2021-06-01", periods=100, freq="ME")

        with pytest.warns(GIPPBreakWarning):
            model = BayesianCausalForest(num_mcmc=10, num_gfr=2)
            model.fit(
                X, data.treatment, data.outcome,
                gipp_date_col="renewal_date"
            )

    def test_no_warn_when_post_gipp(self):
        data = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))
        X = data.X.copy()
        X["renewal_date"] = pd.date_range("2022-03-01", periods=100, freq="ME")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = BayesianCausalForest(num_mcmc=10, num_gfr=2)
            model.fit(X, data.treatment, data.outcome, gipp_date_col="renewal_date")

        gipp_warns = [x for x in w if issubclass(x.category, GIPPBreakWarning)]
        assert len(gipp_warns) == 0


# ------------------------------------------------------------------
# cate
# ------------------------------------------------------------------


class TestCate:
    def test_not_fitted_raises(self):
        model = BayesianCausalForest()
        data = simulate_renewal(SimulationParams(n_policies=50, random_seed=0))
        with pytest.raises(NotFittedError):
            model.cate(data.X)

    def test_returns_dataframe(self, fitted_model, small_dataset):
        result = fitted_model.cate(small_dataset.X)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, fitted_model, small_dataset):
        result = fitted_model.cate(small_dataset.X)
        assert "cate_mean" in result.columns
        assert "cate_lower" in result.columns
        assert "cate_upper" in result.columns
        assert "cate_std" in result.columns

    def test_shape_matches_input(self, fitted_model, small_dataset):
        result = fitted_model.cate(small_dataset.X)
        assert len(result) == len(small_dataset.X)

    def test_index_preserved(self, fitted_model, small_dataset):
        result = fitted_model.cate(small_dataset.X)
        pd.testing.assert_index_equal(result.index, small_dataset.X.index)

    def test_ci_bounds_ordered(self, fitted_model, small_dataset):
        result = fitted_model.cate(small_dataset.X)
        assert (result["cate_lower"] <= result["cate_mean"]).all()
        assert (result["cate_mean"] <= result["cate_upper"]).all()

    def test_credible_level_affects_width(self, fitted_model, small_dataset):
        r90 = fitted_model.cate(small_dataset.X, credible_level=0.90)
        r99 = fitted_model.cate(small_dataset.X, credible_level=0.99)
        width_90 = (r90["cate_upper"] - r90["cate_lower"]).mean()
        width_99 = (r99["cate_upper"] - r99["cate_lower"]).mean()
        assert width_99 >= width_90

    def test_feature_mismatch_raises(self, fitted_model, small_dataset):
        X_bad = small_dataset.X.drop(columns=["age_band"])
        with pytest.raises(ValueError, match="mismatch"):
            fitted_model.cate(X_bad)

    def test_x_nan_raises(self, fitted_model, small_dataset):
        X_nan = small_dataset.X.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fitted_model.cate(X_nan)


# ------------------------------------------------------------------
# posterior_samples
# ------------------------------------------------------------------


class TestPosteriorSamples:
    def test_not_fitted_raises(self):
        model = BayesianCausalForest()
        data = simulate_renewal(SimulationParams(n_policies=50, random_seed=0))
        with pytest.raises(NotFittedError):
            model.posterior_samples(data.X)

    def test_returns_ndarray(self, fitted_model, small_dataset):
        result = fitted_model.posterior_samples(small_dataset.X)
        assert isinstance(result, np.ndarray)

    def test_shape(self, fitted_model, small_dataset):
        result = fitted_model.posterior_samples(small_dataset.X)
        n = len(small_dataset.X)
        assert result.shape[0] == n
        # second dim = n_mcmc samples
        assert result.ndim == 2

    def test_no_nan_in_output(self, fitted_model, small_dataset):
        result = fitted_model.posterior_samples(small_dataset.X)
        assert not np.any(np.isnan(result))


# ------------------------------------------------------------------
# serialisation
# ------------------------------------------------------------------


class TestSerialisation:
    def test_to_json_returns_string(self, fitted_model):
        json_str = fitted_model.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_to_json_unfitted_raises(self):
        model = BayesianCausalForest()
        with pytest.raises(NotFittedError):
            model.to_json()

    def test_from_json_is_fitted(self, fitted_model):
        json_str = fitted_model.to_json()
        model2 = BayesianCausalForest.from_json(json_str, outcome="binary")
        assert model2._is_fitted

    def test_from_json_can_predict(self, fitted_model, small_dataset):
        json_str = fitted_model.to_json()
        model2 = BayesianCausalForest.from_json(json_str, outcome="binary")
        # from_json sets is_fitted but not feature_names — just check no crash
        # (feature validation is skipped if feature_names is empty)
        result = model2.cate(small_dataset.X)
        assert isinstance(result, pd.DataFrame)


# ------------------------------------------------------------------
# propensity score property
# ------------------------------------------------------------------


class TestPropensityScores:
    def test_none_before_fit(self):
        model = BayesianCausalForest()
        assert model.propensity_scores is None

    def test_set_after_fit(self, fitted_model):
        pi = fitted_model.propensity_scores
        assert pi is not None
        assert isinstance(pi, np.ndarray)

    def test_shape_matches_n_train(self, fitted_model, small_dataset):
        pi = fitted_model.propensity_scores
        assert len(pi) == len(small_dataset.X)

    def test_values_in_unit_interval(self, fitted_model):
        pi = fitted_model.propensity_scores
        assert (pi > 0).all()
        assert (pi < 1).all()


# ------------------------------------------------------------------
# convergence summary
# ------------------------------------------------------------------


class TestConvergenceSummary:
    def test_returns_dataframe(self, fitted_model):
        result = fitted_model.convergence_summary()
        assert isinstance(result, pd.DataFrame)

    def test_single_chain_note(self):
        data = simulate_renewal(SimulationParams(n_policies=100, random_seed=0))
        model = BayesianCausalForest(num_mcmc=20, num_gfr=2, num_chains=1)
        model.fit(data.X, data.treatment, data.outcome)
        df = model.convergence_summary()
        assert "num_chains=1" in str(df.values) or "note" in df["parameter"].values
