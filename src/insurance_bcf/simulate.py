"""
Synthetic data generation for insurance BCF scenarios.

All tests and the Databricks demo use synthetic data generated here.
The simulation is designed to produce realistic UK motor insurance renewal
datasets where:
- Treatment effect heterogeneity exists and is recoverable
- Confounding is present (risk score drives both premium and renewal prob)
- Protected characteristic groups have varying lapse sensitivities

The data-generating process is documented so tests can verify that BCF
recovers the known truth within credible intervals.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SimulationParams:
    """Parameters for synthetic motor renewal dataset.

    Attributes
    ----------
    n_policies : int
        Number of policies to simulate. Default 2000.
    n_features : int
        Number of rating factor columns. Minimum 5 (fixed semantics);
        additional columns are pure noise.
    treatment_fraction : float
        Fraction of policies assigned to treatment. Default 0.5.
    base_renewal_prob : float
        Mean renewal probability under control. Default 0.75.
    tau_heterogeneity : float
        Scale of heterogeneous treatment effects. Default 0.08
        (8pp range in lapse sensitivity across segments).
    confounding_strength : float
        Strength of confounding: how much the risk score drives both
        treatment assignment and renewal. Default 1.0 (moderate).
    noise_sigma : float
        Outcome noise standard deviation. Default 0.05.
    random_seed : int
        Random seed. Default 42.
    """

    n_policies: int = 2000
    n_features: int = 10
    treatment_fraction: float = 0.5
    base_renewal_prob: float = 0.75
    tau_heterogeneity: float = 0.08
    confounding_strength: float = 1.0
    noise_sigma: float = 0.05
    random_seed: int = 42


@dataclass
class SimulatedDataset:
    """Output of simulate_renewal().

    Attributes
    ----------
    X : pd.DataFrame
        Covariate matrix with named columns.
    treatment : pd.Series
        Binary treatment indicator (1 = rate increase applied).
    outcome : pd.Series
        Binary renewal flag (1 = renewed).
    true_tau : pd.Series
        Ground-truth CATE for each policy. Useful for validating
        BCF recovery.
    true_propensity : pd.Series
        True P(Z=1 | X). Compare against BCF's estimated propensity.
    params : SimulationParams
        Parameters used to generate this dataset.
    """

    X: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series
    true_tau: pd.Series
    true_propensity: pd.Series
    params: SimulationParams


def simulate_renewal(
    params: SimulationParams | None = None,
    **kwargs: object,
) -> SimulatedDataset:
    """
    Simulate a UK motor insurance renewal dataset for BCF evaluation.

    Data-generating process
    -----------------------
    Features:
      - age_band: integer 0-5 (0=17-24, 1=25-34, ..., 5=65+)
      - ncb_steps: integer 0-5 (No Claims Bonus steps)
      - vehicle_age: integer 0-15 (vehicle age in years)
      - channel: integer 0=direct, 1=PCW, 2=broker
      - policy_duration: float 1-10 (years as customer)
      - [additional noise features if n_features > 5]

    Risk score (confounding):
      risk_score = 0.3*age_band - 0.2*ncb_steps + 0.1*vehicle_age + noise

    Treatment assignment (confounded):
      P(Z=1 | X) = sigmoid(confounding_strength * risk_score)
      High-risk customers are more likely to receive a rate increase.

    True CATE:
      tau(x) = -0.05 - 0.03*(age_band==0) - 0.02*(channel==1) + 0.01*ncb_steps
      Young PCW customers are most lapse-sensitive (-5pp base - 3pp age - 2pp PCW).
      High NCB customers are less sensitive (+1pp per step).

    Outcome:
      mu(x) = base_renewal_prob - 0.05*age_band_young - 0.03*risk_score
      Y_i ~ Bernoulli(sigmoid(mu(x_i) + tau(x_i)*Z_i))

    Parameters
    ----------
    params : SimulationParams or None
        If None, uses defaults.
    **kwargs
        Override any SimulationParams field.

    Returns
    -------
    SimulatedDataset
    """
    if params is None:
        params = SimulationParams(**{k: v for k, v in kwargs.items() if k in SimulationParams.__dataclass_fields__})
    else:
        # Apply overrides
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)

    rng = np.random.default_rng(seed=params.random_seed)
    n = params.n_policies

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    age_band = rng.integers(0, 6, size=n)          # 0=17-24, ..., 5=65+
    ncb_steps = rng.integers(0, 6, size=n)          # 0-5
    vehicle_age = rng.integers(0, 16, size=n)       # 0-15 years
    channel = rng.integers(0, 3, size=n)            # 0=direct,1=PCW,2=broker
    policy_duration = rng.uniform(1, 10, size=n)    # years

    feature_dict: dict[str, npt.NDArray] = {
        "age_band": age_band,
        "ncb_steps": ncb_steps,
        "vehicle_age": vehicle_age,
        "channel": channel,
        "policy_duration": policy_duration,
    }

    # Extra noise features
    for i in range(params.n_features - 5):
        feature_dict[f"noise_{i}"] = rng.standard_normal(n)

    X = pd.DataFrame(feature_dict)

    # ------------------------------------------------------------------
    # Risk score (confounding variable)
    # ------------------------------------------------------------------
    risk_score = (
        0.3 * (age_band - 2.5) / 2.5     # normalised age
        - 0.2 * ncb_steps / 2.5           # lower NCB = higher risk
        + 0.1 * vehicle_age / 7.5         # older vehicle = higher risk
        + rng.normal(0, 0.2, n)
    )

    # ------------------------------------------------------------------
    # Treatment assignment (confounded by risk score)
    # ------------------------------------------------------------------
    log_odds_treat = params.confounding_strength * risk_score
    true_propensity = _sigmoid(log_odds_treat)
    # Clip to avoid extreme values
    true_propensity = np.clip(true_propensity, 0.05, 0.95)
    # Adjust to match treatment_fraction
    target_fraction = params.treatment_fraction
    threshold = np.quantile(true_propensity, 1.0 - target_fraction)
    Z = (true_propensity >= threshold).astype(float)

    # ------------------------------------------------------------------
    # True CATE (heterogeneous treatment effect)
    # ------------------------------------------------------------------
    # Base: 5pp lapse increase for treated group
    # Young (age_band 0) extra -3pp
    # PCW (channel 1) extra -2pp
    # NCB protection: +1pp per step (retained despite lapse sensitivity)
    tau_true = (
        -0.05
        - 0.03 * (age_band == 0).astype(float)
        - 0.02 * (channel == 1).astype(float)
        + 0.01 * ncb_steps
    ) * params.tau_heterogeneity / 0.08  # scale by heterogeneity param

    # ------------------------------------------------------------------
    # Outcome
    # ------------------------------------------------------------------
    # Prognostic: base renewal prob modified by risk
    mu_true = (
        params.base_renewal_prob
        - 0.05 * (age_band == 0).astype(float)    # young => lower base renewal
        - 0.03 * risk_score                        # high risk => lower renewal
    )

    # Linear probability model (not probit) for simplicity in ground truth
    renewal_prob = np.clip(mu_true + tau_true * Z + rng.normal(0, params.noise_sigma, n), 0.01, 0.99)
    outcome = rng.binomial(1, renewal_prob).astype(float)

    return SimulatedDataset(
        X=X,
        treatment=pd.Series(Z, name="treatment"),
        outcome=pd.Series(outcome, name="renewal_flag"),
        true_tau=pd.Series(tau_true, name="true_tau"),
        true_propensity=pd.Series(true_propensity, name="true_propensity"),
        params=params,
    )


def simulate_continuous_outcome(
    n_policies: int = 2000,
    random_seed: int = 42,
) -> SimulatedDataset:
    """
    Simulate a continuous outcome (loss ratio) dataset.

    Treatment is a binary rate increase. Outcome is loss ratio (continuous,
    > 0). Useful for testing outcome='continuous' BCF models.

    Returns
    -------
    SimulatedDataset
        outcome contains loss ratio values (typically 0.3 to 2.0).
    """
    params = SimulationParams(
        n_policies=n_policies,
        random_seed=random_seed,
    )
    data = simulate_renewal(params)

    rng = np.random.default_rng(seed=random_seed + 1)
    # Loss ratio: treatment reduces loss ratio slightly for some segments
    # (e.g. rate increase selects lower-risk renewals)
    loss_ratio = (
        0.80
        - 0.1 * data.true_tau          # effect sign flips for LR
        + rng.normal(0, 0.15, n_policies)
    )
    loss_ratio = np.clip(loss_ratio, 0.01, 5.0)

    return SimulatedDataset(
        X=data.X,
        treatment=data.treatment,
        outcome=pd.Series(loss_ratio, name="loss_ratio"),
        true_tau=-data.true_tau,  # opposite sign for LR
        true_propensity=data.true_propensity,
        params=params,
    )


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))
