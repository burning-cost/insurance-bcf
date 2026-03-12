"""Tests for the MockBCFModel — verifying the mock matches the stochtree API."""
from __future__ import annotations

import numpy as np
import pytest

from insurance_bcf._mock import MockBCFModel


@pytest.fixture()
def fitted_mock():
    model = MockBCFModel()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    Z = rng.integers(0, 2, 100).astype(float)
    y = rng.integers(0, 2, 100).astype(float)
    model.sample(X_train=X, Z_train=Z, y_train=y, num_mcmc=50)
    return model, X, Z, y


class TestMockBCFModelSample:
    def test_is_sampled_after_sample(self):
        model = MockBCFModel()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        Z = rng.integers(0, 2, 50).astype(float)
        y = rng.integers(0, 2, 50).astype(float)
        assert not model.is_sampled()
        model.sample(X, Z, y, num_mcmc=20)
        assert model.is_sampled()

    def test_sample_with_test_data(self):
        model = MockBCFModel()
        rng = np.random.default_rng(1)
        X_tr = rng.standard_normal((50, 4))
        X_te = rng.standard_normal((20, 4))
        Z_tr = rng.integers(0, 2, 50).astype(float)
        Z_te = rng.integers(0, 2, 20).astype(float)
        y_tr = rng.integers(0, 2, 50).astype(float)
        model.sample(X_tr, Z_tr, y_tr, X_test=X_te, Z_test=Z_te, num_mcmc=30)
        assert model.is_sampled()


class TestMockBCFModelPredict:
    def test_returns_dict(self, fitted_mock):
        model, X, Z, y = fitted_mock
        result = model.predict(X, Z)
        assert isinstance(result, dict)

    def test_has_cate_key(self, fitted_mock):
        model, X, Z, y = fitted_mock
        result = model.predict(X, Z, terms="cate")
        assert "cate" in result or "tau" in result

    def test_cate_shape(self, fitted_mock):
        model, X, Z, y = fitted_mock
        result = model.predict(X, Z, terms="cate")
        tau = result.get("cate", result.get("tau"))
        assert tau.shape[0] == len(X)
        assert tau.shape[1] == 50  # num_mcmc


class TestMockBCFModelExtractParameter:
    def test_sigma2(self, fitted_mock):
        model, X, Z, y = fitted_mock
        sigma2 = model.extract_parameter("sigma2")
        assert isinstance(sigma2, np.ndarray)
        assert len(sigma2) > 0

    def test_tau_hat_train(self, fitted_mock):
        model, X, Z, y = fitted_mock
        tau = model.extract_parameter("tau_hat_train")
        assert tau.shape == (100, 50)

    def test_adaptive_coding(self, fitted_mock):
        model, X, Z, y = fitted_mock
        ac = model.extract_parameter("adaptive_coding")
        assert isinstance(ac, np.ndarray)


class TestMockBCFModelSerialisation:
    def test_to_json_returns_string(self, fitted_mock):
        model, X, Z, y = fitted_mock
        json_str = model.to_json()
        assert isinstance(json_str, str)

    def test_from_json_is_sampled(self, fitted_mock):
        model, X, Z, y = fitted_mock
        json_str = model.to_json()
        model2 = MockBCFModel.from_json(json_str)
        assert model2.is_sampled()

    def test_has_term(self, fitted_mock):
        model, X, Z, y = fitted_mock
        assert model.has_term("sigma2")
        assert model.has_term("tau_hat_train")
        assert not model.has_term("nonexistent")


class TestMockBCFModelComputePosteriorInterval:
    def test_returns_dict(self, fitted_mock):
        model, X, Z, y = fitted_mock
        result = model.compute_posterior_interval(X=X, Z=Z)
        assert isinstance(result, dict)

    def test_has_ci_keys(self, fitted_mock):
        model, X, Z, y = fitted_mock
        result = model.compute_posterior_interval(X=X, Z=Z)
        # Should have lower/upper for cate or tau
        has_lower = any("lower" in k for k in result.keys())
        has_upper = any("upper" in k for k in result.keys())
        assert has_lower
        assert has_upper
