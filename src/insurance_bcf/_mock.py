"""
Mock BCFModel for environments where stochtree cannot install.

stochtree requires a C++ backend compiled for the target architecture.
On ARM64 (Raspberry Pi) and some CI environments this fails. This module
provides a drop-in mock that passes the correct interface so tests can
verify the insurance_bcf layer without the C++ dependency.

The mock returns plausible-shaped outputs using simple linear combinations
of the input data — enough to exercise every code path without real MCMC.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class MockBCFModel:
    """
    Minimal BCFModel stand-in. Matches the stochtree.bcf.BCFModel interface
    used by insurance_bcf — not the full stochtree API.

    Only the methods called by BayesianCausalForest are implemented.
    """

    def __init__(self) -> None:
        self._sampled = False
        self._n_mcmc = 0
        self._rng = np.random.default_rng(seed=0)
        # Parameters set during sample()
        self._n_train: int = 0
        self._n_test: int = 0
        self._tau_train: npt.NDArray[np.float64] | None = None
        self._tau_test: npt.NDArray[np.float64] | None = None
        self._y_hat_train: npt.NDArray[np.float64] | None = None
        self._y_hat_test: npt.NDArray[np.float64] | None = None
        self._sigma2: npt.NDArray[np.float64] | None = None
        self._json_str: str = ""

    # ------------------------------------------------------------------
    # sample — mimics model.sample(X_train, Z_train, y_train, ...)
    # ------------------------------------------------------------------

    def sample(
        self,
        X_train: npt.ArrayLike,
        Z_train: npt.ArrayLike,
        y_train: npt.ArrayLike,
        propensity_train: npt.ArrayLike | None = None,
        rfx_group_ids_train: npt.ArrayLike | None = None,
        rfx_basis_train: npt.ArrayLike | None = None,
        X_test: npt.ArrayLike | None = None,
        Z_test: npt.ArrayLike | None = None,
        propensity_test: npt.ArrayLike | None = None,
        rfx_group_ids_test: npt.ArrayLike | None = None,
        rfx_basis_test: npt.ArrayLike | None = None,
        num_gfr: int = 5,
        num_burnin: int = 0,
        num_mcmc: int = 100,
        previous_model_json: str | None = None,
        previous_model_warmstart_sample_num: int | None = None,
        general_params: dict | None = None,
        prognostic_forest_params: dict | None = None,
        treatment_effect_forest_params: dict | None = None,
        variance_forest_params: dict | None = None,
        random_effects_params: dict | None = None,
    ) -> None:
        X_arr = np.asarray(X_train, dtype=float)
        Z_arr = np.asarray(Z_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)

        n_train = X_arr.shape[0]
        self._n_train = n_train
        self._n_mcmc = num_mcmc

        # Generate synthetic tau: weak linear signal on first feature + noise
        rng = self._rng
        signal = 0.05 * (X_arr[:, 0] - X_arr[:, 0].mean()) if X_arr.shape[1] > 0 else np.zeros(n_train)
        tau_mean = signal[:, None] + rng.normal(0, 0.02, (n_train, num_mcmc))
        self._tau_train = tau_mean

        mu_mean = y_arr[:, None] - tau_mean * Z_arr[:, None] + rng.normal(0, 0.01, (n_train, num_mcmc))
        self._y_hat_train = mu_mean + tau_mean * Z_arr[:, None]

        self._sigma2 = np.abs(rng.normal(0.1, 0.02, num_mcmc))

        if X_test is not None:
            X_test_arr = np.asarray(X_test, dtype=float)
            n_test = X_test_arr.shape[0]
            self._n_test = n_test
            signal_test = 0.05 * (X_test_arr[:, 0] - X_arr[:, 0].mean()) if X_test_arr.shape[1] > 0 else np.zeros(n_test)
            self._tau_test = signal_test[:, None] + rng.normal(0, 0.02, (n_test, num_mcmc))
            Z_test_arr = np.asarray(Z_test, dtype=float) if Z_test is not None else np.ones(n_test)
            self._y_hat_test = rng.normal(0, 0.1, (n_test, num_mcmc)) + self._tau_test * Z_test_arr[:, None]
        else:
            self._n_test = 0
            self._tau_test = None
            self._y_hat_test = None

        self._sampled = True
        self._json_str = '{"mock": true}'

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: npt.ArrayLike,
        Z: npt.ArrayLike,
        propensity: npt.ArrayLike | None = None,
        rfx_group_ids: npt.ArrayLike | None = None,
        rfx_basis: npt.ArrayLike | None = None,
        type: str = "posterior",
        terms: str = "all",
        scale: str = "linear",
    ) -> dict[str, npt.NDArray[np.float64]]:
        X_arr = np.asarray(X, dtype=float)
        Z_arr = np.asarray(Z, dtype=float)
        n = X_arr.shape[0]

        signal = 0.05 * (X_arr[:, 0] - X_arr[:, 0].mean()) if X_arr.shape[1] > 0 else np.zeros(n)
        tau = signal[:, None] + self._rng.normal(0, 0.02, (n, self._n_mcmc))
        mu = self._rng.normal(0.5, 0.05, (n, self._n_mcmc))
        y_hat = mu + tau * Z_arr[:, None]

        return {
            "y_hat": y_hat,
            "prognostic_function": mu,
            "mu": mu,
            "cate": tau,
            "tau": tau,
        }

    # ------------------------------------------------------------------
    # compute_posterior_interval
    # ------------------------------------------------------------------

    def compute_posterior_interval(
        self,
        X: npt.ArrayLike | None = None,
        Z: npt.ArrayLike | None = None,
        propensity: npt.ArrayLike | None = None,
        rfx_group_ids: npt.ArrayLike | None = None,
        rfx_basis: npt.ArrayLike | None = None,
        terms: str = "all",
        level: float = 0.95,
        scale: str = "linear",
    ) -> dict[str, npt.NDArray[np.float64]]:
        assert X is not None
        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        signal = 0.05 * (X_arr[:, 0] - X_arr[:, 0].mean()) if X_arr.shape[1] > 0 else np.zeros(n)
        alpha = (1.0 - level) / 2.0
        noise = 0.03
        lower = signal - noise
        upper = signal + noise
        mean_ = signal
        return {
            "cate_lower": lower,
            "cate_upper": upper,
            "cate_mean": mean_,
            "tau_lower": lower,
            "tau_upper": upper,
            "tau_mean": mean_,
        }

    # ------------------------------------------------------------------
    # extract_parameter
    # ------------------------------------------------------------------

    def extract_parameter(self, term: str) -> npt.NDArray[np.float64]:
        if term == "sigma2":
            return self._sigma2 if self._sigma2 is not None else np.array([0.1])
        if term in ("tau_hat_train",) and self._tau_train is not None:
            return self._tau_train
        if term in ("tau_hat_test",) and self._tau_test is not None:
            return self._tau_test
        if term in ("y_hat_train",) and self._y_hat_train is not None:
            return self._y_hat_train
        if term in ("y_hat_test",) and self._y_hat_test is not None:
            return self._y_hat_test
        if term == "adaptive_coding":
            return np.array([0.0, 1.0])
        # sigma2_leaf_mu / sigma2_leaf_tau
        return np.abs(self._rng.normal(0.05, 0.01, self._n_mcmc if self._n_mcmc > 0 else 10))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return self._json_str

    @classmethod
    def from_json(cls, json_str: str) -> "MockBCFModel":
        obj = cls()
        obj._json_str = json_str
        obj._sampled = True
        obj._n_mcmc = 100
        obj._sigma2 = np.abs(np.random.normal(0.1, 0.02, 100))
        return obj

    def is_sampled(self) -> bool:
        return self._sampled

    def has_term(self, term: str) -> bool:
        return term in ("sigma2", "tau_hat_train", "tau_hat_test", "y_hat_train", "y_hat_test", "adaptive_coding", "sigma2_leaf_mu", "sigma2_leaf_tau")


def _get_bcf_model_class() -> type:
    """
    Return BCFModel from stochtree if available, else the mock.

    Import is deferred so that importing insurance_bcf does not crash
    on machines without the C++ stochtree wheel installed.
    """
    try:
        from stochtree.bcf import BCFModel  # type: ignore[import-untyped]
        return BCFModel
    except ImportError:
        import warnings
        warnings.warn(
            "stochtree is not installed or could not be imported. "
            "Falling back to MockBCFModel — results are NOT real MCMC. "
            "Install stochtree>=0.4.0 for production use: pip install stochtree",
            ImportWarning,
            stacklevel=3,
        )
        return MockBCFModel
