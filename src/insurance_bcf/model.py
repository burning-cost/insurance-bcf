"""
BayesianCausalForest — insurance wrapper over stochtree BCFModel.

The model equation:

    Y_i = mu(x_i, pi_hat(x_i)) + tau(x_i) * z_i + epsilon_i

where mu is the prognostic function (250 trees, expressive prior) and tau is
the treatment effect function (50 trees, shrink-to-homogeneity prior).

Including pi_hat in mu corrects Regularization-Induced Confounding (RIC) —
the mechanism by which BART over-shrinks mu and incorrectly attributes
unexplained variance to the treatment effect. This is the central BCF
innovation from Hahn, Murray, Carvalho (2020) Bayesian Analysis 15(3).

Reference: Herren et al. (2025/2026) arXiv:2512.12051v2 — stochtree paper.

stochtree API notes (v0.4.0):
- BCFModel.predict() requires propensity if propensity was passed to sample().
  We always pass propensity to sample(), so we must pass it to predict() too.
  For predict, we use the training propensity (mean) as a reasonable default
  when predicting on training data, or None if the user can provide their own.
- BCFModel.from_json() takes json_string= as a keyword argument.
- rfx_group_ids_train requires rfx_basis_train — we don't support random effects
  without a basis matrix. The groups parameter is reserved for future use.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ._mock import _get_bcf_model_class


# ------------------------------------------------------------------
# Custom warnings and exceptions
# ------------------------------------------------------------------


class GIPPBreakWarning(UserWarning):
    """
    Raised when the dataset appears to span the January 2022 GIPP remedy date.

    FCA General Insurance Pricing Practices (GIPP) rules came into force in
    January 2022 and changed renewal pricing materially. A model fitted across
    this break conflates pre-GIPP and post-GIPP pricing behaviour. Always
    confirm the data window is homogeneous with respect to this regulatory event.
    """


class PositivityViolationError(ValueError):
    """
    Raised when propensity scores violate the overlap assumption.

    BCF (and all causal inference methods based on unconfoundedness) requires
    that every unit has some probability of receiving either treatment:
        0 < P(Z=1 | X) < 1 for all X in the support.

    If propensity < 0.05 or > 0.95 for more than 5% of units, the treatment
    effect is estimated by extrapolation, not interpolation. The estimates are
    unreliable. Trim the support or use a different estimator.
    """


class ConvergenceWarning(UserWarning):
    """
    Raised when R-hat > 1.1 for any monitored parameter.

    R-hat (potential scale reduction factor) compares within-chain to
    between-chain variance across parallel MCMC chains. Values > 1.1 indicate
    the chains have not converged to the same posterior. Increase num_mcmc,
    increase num_chains, or inspect traceplots before trusting results.

    Only computed when num_chains > 1 (requires arviz).
    """


class NotFittedError(RuntimeError):
    """Raised when predict/cate is called before fit()."""


# ------------------------------------------------------------------
# BayesianCausalForest
# ------------------------------------------------------------------


class BayesianCausalForest:
    """
    Bayesian Causal Forests for heterogeneous treatment effect estimation.

    A thin but opinionated wrapper over stochtree's BCFModel with insurance
    defaults, propensity estimation, positivity diagnostics, and convergence
    monitoring.

    The library is designed around the UK renewal pricing use case:
    - Y: binary renewal flag or continuous loss ratio
    - Z: binary treatment (rate increase applied, telematics policy, NCD change)
    - X: rating factors (age band, NCB steps, vehicle age, channel, postcode)

    Parameters
    ----------
    outcome : str
        'binary' for renewal flags (activates probit link inside stochtree).
        'continuous' for loss ratios, claim counts, or any real-valued outcome.
    treatment_trees : int
        Number of trees in the tau forest. Default 50 — the stochtree default
        from Hahn et al. (2020). The shrink-to-homogeneity prior on tau works
        best with 50 trees; increasing this is not recommended without testing.
    prognostic_trees : int
        Number of trees in the mu forest. Default 250. The prognostic function
        benefits from more capacity because it models the full outcome surface.
    num_mcmc : int
        Number of retained posterior samples after GFR warm-start. Default 500
        (stochtree uses 100; 500 gives smoother credible intervals for
        audit-grade reporting).
    num_gfr : int
        Number of Grow-From-Root warm-start iterations (He & Hahn 2021).
        Eliminates the need for long burn-in. Default 10.
    num_burnin : int
        Standard MCMC burn-in after GFR. Usually 0 when num_gfr > 0.
    num_chains : int
        Number of parallel MCMC chains. Default 1. Set to 4 for R-hat
        convergence diagnostics (requires arviz).
    propensity_covariate : str
        Where to include pi_hat in the BCF model. 'prognostic' (default) adds
        it to the mu forest only — this is the correct RIC-correcting
        specification. Never set to 'none' for insurance observational data.
    random_seed : int or None
        Seed for reproducibility.
    positivity_threshold : float
        Propensity scores outside [positivity_threshold, 1 - positivity_threshold]
        trigger a PositivityViolationError. Default 0.05.
    positivity_max_fraction : float
        Maximum fraction of units allowed to violate the positivity threshold
        before raising. Default 0.05 (5%).
    """

    def __init__(
        self,
        outcome: Literal["binary", "continuous"] = "binary",
        treatment_trees: int = 50,
        prognostic_trees: int = 250,
        num_mcmc: int = 500,
        num_gfr: int = 10,
        num_burnin: int = 0,
        num_chains: int = 1,
        propensity_covariate: Literal[
            "prognostic", "treatment_effect", "both", "none"
        ] = "prognostic",
        random_seed: int | None = None,
        positivity_threshold: float = 0.05,
        positivity_max_fraction: float = 0.05,
    ) -> None:
        if outcome not in ("binary", "continuous"):
            raise ValueError(f"outcome must be 'binary' or 'continuous', got {outcome!r}")
        if propensity_covariate == "none":
            warnings.warn(
                "propensity_covariate='none' disables RIC correction. "
                "This is not recommended for insurance observational data where "
                "risk scores drive both premium assignment and renewal probability.",
                UserWarning,
                stacklevel=2,
            )

        self.outcome = outcome
        self.treatment_trees = treatment_trees
        self.prognostic_trees = prognostic_trees
        self.num_mcmc = num_mcmc
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_chains = num_chains
        self.propensity_covariate = propensity_covariate
        self.random_seed = random_seed
        self.positivity_threshold = positivity_threshold
        self.positivity_max_fraction = positivity_max_fraction

        self._bcf_model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._propensity_train: npt.NDArray[np.float64] | None = None
        self._n_train: int = 0
        self._BCFModelClass = _get_bcf_model_class()
        # Track whether we're using the mock (for API routing)
        self._using_mock: bool = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        propensity: npt.ArrayLike | None = None,
        groups: pd.Series | None = None,
        gipp_date_col: str | None = None,
    ) -> "BayesianCausalForest":
        """
        Fit the BCF model.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix. All columns must be numeric (one-hot or ordinal
            encoded). Shape (n_policies, n_features).
        treatment : pd.Series
            Binary or continuous treatment indicator. Binary is the primary use
            case: 1 = rate increase applied, 0 = not applied.
        outcome : pd.Series
            Outcome variable. Binary for renewal flag (0/1), continuous for
            loss ratio or claim cost.
        propensity : array-like or None
            Pre-computed propensity scores P(Z=1 | X). Shape (n_policies,).
            If None, estimated from X using logistic regression. External
            propensity is preferred for insurance applications where you can
            encode domain knowledge about treatment assignment.
        groups : pd.Series or None
            Reserved for future random effects support. Currently unused in
            the stochtree call (requires rfx_basis_train which we don't support).
        gipp_date_col : str or None
            If provided, the name of a date column in X to check for GIPP break.
            Issues GIPPBreakWarning if data spans January 2022.

        Returns
        -------
        self : BayesianCausalForest
            Fitted model (for method chaining).
        """
        X_arr, Z_arr, y_arr = self._validate_inputs(X, treatment, outcome)

        # GIPP break check
        if gipp_date_col is not None:
            self._check_gipp_break(X, gipp_date_col)

        # Propensity estimation
        pi_hat = self._resolve_propensity(X_arr, Z_arr, propensity)
        self._positivity_check(pi_hat)
        self._propensity_train = pi_hat

        # Build stochtree params
        general_params: dict = {
            "propensity_covariate": self.propensity_covariate,
            "adaptive_coding": True,
            "num_chains": self.num_chains,
            "standardize": True,
            "sample_sigma2_global": True,
        }
        if self.outcome == "binary":
            general_params["probit_outcome_model"] = True
        if self.random_seed is not None:
            general_params["random_seed"] = self.random_seed

        prognostic_params: dict = {
            "num_trees": self.prognostic_trees,
            "alpha": 0.95,
            "beta": 2,
            "max_depth": 10,
        }
        treatment_params: dict = {
            "num_trees": self.treatment_trees,
            "alpha": 0.25,
            "beta": 3,
            "max_depth": 5,
        }

        # Determine if we're using real stochtree or mock
        BCFModelClass = self._BCFModelClass
        try:
            from stochtree.bcf import BCFModel as _RealBCFModel  # type: ignore[import-untyped]
            self._using_mock = BCFModelClass is not _RealBCFModel
        except ImportError:
            self._using_mock = True

        # Instantiate and sample
        self._bcf_model = BCFModelClass()
        sample_kwargs: dict = dict(
            X_train=X_arr,
            Z_train=Z_arr,
            y_train=y_arr,
            propensity_train=pi_hat,
            num_gfr=self.num_gfr,
            num_burnin=self.num_burnin,
            num_mcmc=self.num_mcmc,
            general_params=general_params,
            prognostic_forest_params=prognostic_params,
            treatment_effect_forest_params=treatment_params,
        )
        # Note: rfx_group_ids_train requires rfx_basis_train in stochtree 0.4.0
        # Random effects are not passed until we support the full basis interface

        self._bcf_model.sample(**sample_kwargs)

        self._feature_names = list(X.columns)
        self._n_train = len(y_arr)
        self._is_fitted = True

        # Convergence diagnostics (best-effort, requires arviz + num_chains > 1)
        if self.num_chains > 1:
            self._check_convergence()

        return self

    # ------------------------------------------------------------------
    # cate — point estimates + credible intervals
    # ------------------------------------------------------------------

    def cate(
        self,
        X: pd.DataFrame,
        treatment_value: float = 1.0,
        credible_level: float = 0.95,
        propensity: npt.ArrayLike | None = None,
    ) -> pd.DataFrame:
        """
        Compute posterior CATE estimates with credible intervals.

        For binary outcome (probit link), tau is on the latent normal scale.
        Use `marginalise_probit=True` in posterior_samples() for probability
        scale effects.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix, same columns as fit(). Shape (n, p).
        treatment_value : float
            Treatment level for which to compute CATE. Default 1.0 (binary).
        credible_level : float
            Posterior credible interval level. Default 0.95.
        propensity : array-like or None
            Propensity scores for X. If None and using real stochtree,
            falls back to mean training propensity (constant). Provide
            for more accurate predictions on new data.

        Returns
        -------
        pd.DataFrame
            Columns: cate_mean, cate_lower, cate_upper, cate_std.
            Shape (n, 4). Index matches X.
        """
        self._check_fitted()
        X_arr = self._validate_features(X)
        Z_arr = np.full(len(X_arr), treatment_value)

        tau_posterior = self._get_tau_posterior(X_arr, Z_arr, propensity)

        alpha = (1.0 - credible_level) / 2.0
        return pd.DataFrame(
            {
                "cate_mean": tau_posterior.mean(axis=1),
                "cate_lower": np.quantile(tau_posterior, alpha, axis=1),
                "cate_upper": np.quantile(tau_posterior, 1.0 - alpha, axis=1),
                "cate_std": tau_posterior.std(axis=1),
            },
            index=X.index,
        )

    # ------------------------------------------------------------------
    # posterior_samples — full posterior over tau
    # ------------------------------------------------------------------

    def posterior_samples(
        self,
        X: pd.DataFrame,
        treatment_value: float = 1.0,
        marginalise_probit: bool = False,
        propensity: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Return raw posterior draws over tau(x).

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        treatment_value : float
            Treatment level. Default 1.0.
        marginalise_probit : bool
            If True and outcome=='binary', apply the standard normal CDF to
            convert latent probit tau to approximate probability-scale effects.
            Default False (return latent scale).
        propensity : array-like or None
            Propensity scores for X. If None, uses mean training propensity.

        Returns
        -------
        np.ndarray
            Shape (n_obs, num_mcmc_samples).
        """
        self._check_fitted()
        X_arr = self._validate_features(X)
        Z_arr = np.full(len(X_arr), treatment_value)

        tau_posterior = self._get_tau_posterior(X_arr, Z_arr, propensity)

        if marginalise_probit and self.outcome == "binary":
            from scipy.stats import norm  # type: ignore[import-untyped]
            tau_posterior = norm.cdf(tau_posterior)

        return np.asarray(tau_posterior, dtype=float)

    # ------------------------------------------------------------------
    # Internal: get tau posterior from stochtree
    # ------------------------------------------------------------------

    def _get_tau_posterior(
        self,
        X_arr: npt.NDArray[np.float64],
        Z_arr: npt.NDArray[np.float64],
        propensity: npt.ArrayLike | None,
    ) -> npt.NDArray[np.float64]:
        """
        Call stochtree predict() and extract tau posterior.

        stochtree 0.4.0 requires propensity to be passed to predict() when
        it was passed to sample(). We always pass propensity to sample(), so
        we must always pass it here.

        When predicting on new data without user-supplied propensity, we use
        the mean training propensity as a constant. This is a simplification —
        for production use, compute an appropriate propensity for the new data.
        """
        if self._using_mock:
            # Mock doesn't need propensity in predict
            preds = self._bcf_model.predict(X=X_arr, Z=Z_arr, type="posterior", terms="cate")
        else:
            # Real stochtree: must pass propensity
            if propensity is not None:
                pi = np.asarray(propensity, dtype=float)
            elif self._propensity_train is not None:
                # Use mean training propensity as constant for new data
                pi = np.full(len(X_arr), float(self._propensity_train.mean()))
            else:
                pi = np.full(len(X_arr), 0.5)

            try:
                preds = self._bcf_model.predict(
                    X=X_arr,
                    Z=Z_arr,
                    propensity=pi,
                    type="posterior",
                    terms="cate",
                )
            except TypeError:
                # Fallback: older stochtree may not accept propensity in predict
                preds = self._bcf_model.predict(X=X_arr, Z=Z_arr, type="posterior", terms="cate")

        # stochtree returns dict with 'cate' key when terms='cate'
        # Shape: (n, n_mcmc_samples)
        if isinstance(preds, dict):
            tau_posterior = preds.get("cate", preds.get("tau", None))
        else:
            tau_posterior = preds

        if tau_posterior is None:
            raise RuntimeError(
                "stochtree predict() did not return CATE samples. "
                "Check the stochtree version and terms parameter."
            )

        return np.asarray(tau_posterior, dtype=float)

    # ------------------------------------------------------------------
    # propensity — return propensity scores used in fit
    # ------------------------------------------------------------------

    @property
    def propensity_scores(self) -> npt.NDArray[np.float64] | None:
        """Propensity scores used during fit(). Shape (n_train,)."""
        return self._propensity_train

    # ------------------------------------------------------------------
    # convergence_summary
    # ------------------------------------------------------------------

    def convergence_summary(self) -> pd.DataFrame:
        """
        Return R-hat statistics for key parameters (requires arviz, num_chains>1).

        Returns a DataFrame with columns: parameter, r_hat, converged.
        If arviz is not installed or num_chains==1, returns an empty DataFrame
        with a note column.
        """
        self._check_fitted()
        if self.num_chains < 2:
            return pd.DataFrame(
                [{"parameter": "note", "r_hat": float("nan"), "converged": None,
                  "detail": "num_chains=1; increase to 4 for R-hat diagnostics"}]
            )
        try:
            import arviz as az  # type: ignore[import-untyped]
        except ImportError:
            return pd.DataFrame(
                [{"parameter": "note", "r_hat": float("nan"), "converged": None,
                  "detail": "arviz not installed; pip install arviz for R-hat"}]
            )

        records = []
        for term in ("sigma2",):
            try:
                samples = self._bcf_model.extract_parameter(term)
                # samples shape: (n_mcmc,) or (n_chains, n_mcmc)
                if samples.ndim == 1:
                    # Treat as single chain — can't compute R-hat
                    records.append({"parameter": term, "r_hat": float("nan"), "converged": None,
                                    "detail": "single-chain samples; no R-hat"})
                else:
                    data = az.from_dict({"posterior": {term: samples[None, :]}})
                    rhat = az.rhat(data)["posterior"][term].values.item()
                    records.append({"parameter": term, "r_hat": rhat, "converged": rhat <= 1.1})
            except Exception as exc:
                records.append({"parameter": term, "r_hat": float("nan"), "converged": None,
                                "detail": str(exc)})

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the fitted BCF model to JSON string."""
        self._check_fitted()
        return self._bcf_model.to_json()  # type: ignore[union-attr]

    @classmethod
    def from_json(
        cls,
        json_str: str,
        outcome: Literal["binary", "continuous"] = "binary",
        **kwargs: object,
    ) -> "BayesianCausalForest":
        """
        Deserialise a fitted BCF model from JSON string.

        Parameters
        ----------
        json_str : str
            Output of a previous to_json() call.
        outcome : str
            'binary' or 'continuous' — must match the original fit.
        **kwargs
            Any constructor kwargs to restore on the wrapper.
        """
        BCFModelClass = _get_bcf_model_class()
        # stochtree 0.4.0 BCFModel.from_json() is an instance method:
        #   instance = BCFModel()
        #   instance.from_json(json_string=json_str)
        # This is unusual but reflects the stochtree implementation.
        # Try multiple calling conventions for robustness.
        instance = BCFModelClass()
        try:
            # Instance method with keyword arg (stochtree 0.4.0 real API)
            instance.from_json(json_string=json_str)
            bcf_model = instance
        except TypeError:
            try:
                # Instance method with positional arg
                instance.from_json(json_str)
                bcf_model = instance
            except TypeError:
                try:
                    # Class/static method with keyword arg
                    bcf_model = BCFModelClass.from_json(json_string=json_str)
                except TypeError:
                    # Class/static method with positional arg (mock)
                    bcf_model = BCFModelClass.from_json(json_str)
        obj = cls(outcome=outcome, **kwargs)
        obj._bcf_model = bcf_model
        obj._is_fitted = True
        # Determine mock status
        try:
            from stochtree.bcf import BCFModel as _RealBCFModel  # type: ignore[import-untyped]
            obj._using_mock = BCFModelClass is not _RealBCFModel
        except ImportError:
            obj._using_mock = True
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame, got {type(X).__name__}")
        if len(X) == 0:
            raise ValueError("X must not be empty")
        if X.isnull().any().any():
            raise ValueError(
                "X contains NaN values. Impute missing values before fitting BCF."
            )
        if len(treatment) != len(X):
            raise ValueError(
                f"treatment length {len(treatment)} != X rows {len(X)}"
            )
        if len(outcome) != len(X):
            raise ValueError(
                f"outcome length {len(outcome)} != X rows {len(X)}"
            )

        Z = pd.Series(treatment).to_numpy(dtype=float)
        y = pd.Series(outcome).to_numpy(dtype=float)

        if np.any(np.isnan(Z)):
            raise ValueError("treatment contains NaN values")
        if np.any(np.isnan(y)):
            raise ValueError("outcome contains NaN values")

        if self.outcome == "binary":
            unique_vals = set(np.unique(y))
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"outcome='binary' but y contains non-binary values: {unique_vals}. "
                    "Use outcome='continuous' for non-binary outcomes."
                )

        X_arr = X.to_numpy(dtype=float)
        return X_arr, Z, y

    def _validate_features(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pd.DataFrame, got {type(X).__name__}")
        if self._feature_names and list(X.columns) != self._feature_names:
            missing = set(self._feature_names) - set(X.columns)
            extra = set(X.columns) - set(self._feature_names)
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            raise ValueError(
                f"Feature mismatch vs. fit(): {'; '.join(parts)}. "
                "Use the same columns as passed to fit()."
            )
        if X.isnull().any().any():
            raise ValueError("X contains NaN values.")
        return X.to_numpy(dtype=float)

    def _resolve_propensity(
        self,
        X_arr: npt.NDArray[np.float64],
        Z_arr: npt.NDArray[np.float64],
        propensity: npt.ArrayLike | None,
    ) -> npt.NDArray[np.float64]:
        if propensity is not None:
            pi = np.asarray(propensity, dtype=float)
            if pi.shape != (len(Z_arr),):
                raise ValueError(
                    f"propensity shape {pi.shape} does not match n_policies={len(Z_arr)}"
                )
            if np.any(pi <= 0) or np.any(pi >= 1):
                raise ValueError(
                    "Propensity scores must be strictly in (0, 1). "
                    "Values <= 0 or >= 1 are not valid probabilities."
                )
            return pi

        # Estimate via logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        # Detect binary vs continuous treatment for propensity model
        unique_z = np.unique(Z_arr)
        if len(unique_z) == 2 and set(unique_z).issubset({0, 1, 0.0, 1.0}):
            # Binary treatment: logistic regression
            lr = LogisticRegression(
                max_iter=500,
                C=1.0,
                solver="lbfgs",
                random_state=self.random_seed,
            )
            lr.fit(X_scaled, Z_arr.astype(int))
            pi = lr.predict_proba(X_scaled)[:, 1].astype(float)
        else:
            # Continuous treatment: use Gaussian approximation via OLS residuals
            from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_scaled, Z_arr)
            z_pred = ridge.predict(X_scaled)
            residuals = Z_arr - z_pred
            sigma = np.std(residuals) + 1e-8
            from scipy.stats import norm  # type: ignore[import-untyped]
            z_median = np.median(Z_arr)
            pi = norm.cdf((z_pred - z_median) / sigma)
            pi = np.clip(pi, 0.01, 0.99)

        return pi

    def _positivity_check(self, pi: npt.NDArray[np.float64]) -> None:
        lo = self.positivity_threshold
        hi = 1.0 - self.positivity_threshold
        violating = np.mean((pi < lo) | (pi > hi))
        if violating > self.positivity_max_fraction:
            raise PositivityViolationError(
                f"{violating:.1%} of policies have propensity scores outside "
                f"[{lo}, {hi}]. The overlap assumption is violated for a substantial "
                f"fraction of the portfolio. Treatment effect estimates will rely on "
                f"extrapolation and cannot be trusted. "
                f"Consider: (1) trimming policies with extreme propensities, "
                f"(2) refining the treatment assignment mechanism, or "
                f"(3) using matching/trimming before fitting."
            )
        # Warn for borderline cases
        warn_count = np.sum((pi < lo) | (pi > hi))
        if warn_count > 0:
            warnings.warn(
                f"{warn_count} policies ({warn_count/len(pi):.1%}) have propensity "
                f"scores outside [{lo}, {hi}] but below the {self.positivity_max_fraction:.0%} "
                f"threshold. Monitor overlap carefully.",
                UserWarning,
                stacklevel=4,
            )

    def _check_gipp_break(self, X: pd.DataFrame, date_col: str) -> None:
        if date_col not in X.columns:
            return
        try:
            dates = pd.to_datetime(X[date_col])
            gipp_date = pd.Timestamp("2022-01-01")
            if dates.min() < gipp_date <= dates.max():
                warnings.warn(
                    f"Column '{date_col}' spans the GIPP implementation date (January 2022). "
                    "FCA General Insurance Pricing Practices rules changed renewal pricing "
                    "materially at this date. Mixing pre-GIPP and post-GIPP data conflates "
                    "two distinct pricing regimes. Consider filtering to one period.",
                    GIPPBreakWarning,
                    stacklevel=4,
                )
        except Exception:
            pass  # Don't crash on unparseable dates — just skip the check

    def _check_convergence(self) -> None:
        try:
            import arviz as az  # type: ignore[import-untyped]
        except ImportError:
            return

        try:
            sigma2 = self._bcf_model.extract_parameter("sigma2")  # type: ignore[union-attr]
            if sigma2.ndim >= 1 and len(sigma2) > 1:
                data = az.from_dict({"posterior": {"sigma2": sigma2[None, :]}})
                rhat = az.rhat(data)["posterior"]["sigma2"].values.item()
                if rhat > 1.1:
                    warnings.warn(
                        f"R-hat for sigma2 = {rhat:.3f} > 1.1. MCMC chains may not have "
                        "converged. Increase num_mcmc or inspect traceplots via "
                        "convergence_summary().",
                        ConvergenceWarning,
                        stacklevel=4,
                    )
        except Exception:
            pass

    def _check_fitted(self) -> None:
        if not self._is_fitted or self._bcf_model is None:
            raise NotFittedError(
                "Model is not fitted. Call fit() before predict() or cate()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"BayesianCausalForest("
            f"outcome={self.outcome!r}, "
            f"treatment_trees={self.treatment_trees}, "
            f"prognostic_trees={self.prognostic_trees}, "
            f"num_mcmc={self.num_mcmc}, "
            f"status={status!r})"
        )
