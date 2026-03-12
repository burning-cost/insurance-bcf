"""
insurance-bcf: Bayesian Causal Forests for insurance pricing.

The problem this solves
-----------------------
UK motor insurers applying rate changes want to know: which customer segments
lapse most in response to a rate increase? A GLM gives you one number — the
average. BCF gives you a posterior distribution over the treatment effect for
every policy in the portfolio.

The method — BCF (Bayesian Causal Forests, Hahn Murray Carvalho 2020) — runs
two separate Bayesian tree ensembles:

    Y_i = mu(x_i, pi_hat(x_i)) + tau(x_i) * z_i + epsilon_i

mu captures the renewal probability surface under control. tau captures
treatment effect heterogeneity. The priors are deliberately asymmetric: tau
is shrunk hard toward homogeneity (alpha=0.25, beta=3 vs. alpha=0.95, beta=2
for mu) unless the data strongly supports differential effects.

Including pi_hat (propensity score) in mu corrects Regularization-Induced
Confounding — the mechanism by which standard BART incorrectly attributes
unexplained outcome variance to the treatment when the two are correlated
through the risk model. This is not an optional extra — it is essential for
observational insurance renewal data.

Quick start
-----------
>>> from insurance_bcf import BayesianCausalForest, ElasticityEstimator
>>> from insurance_bcf.simulate import simulate_renewal
>>>
>>> data = simulate_renewal(n_policies=5000)
>>> model = BayesianCausalForest(outcome='binary', num_mcmc=200)
>>> model.fit(data.X, data.treatment, data.outcome)
>>>
>>> cate_df = model.cate(data.X)
>>> print(cate_df[['cate_mean', 'cate_lower', 'cate_upper']].head())
>>>
>>> est = ElasticityEstimator(model)
>>> seg = est.segment_effects(data.X, segment_cols=['age_band', 'channel'])
>>> print(seg)

FCA EP25/2 audit report
-----------------------
>>> from insurance_bcf import BCFAuditReport
>>> report = BCFAuditReport(model, est)
>>> report.render(
...     'bcf_audit_2024Q4.html',
...     X=data.X,
...     Z=data.treatment,
...     protected_cols=['age_band'],
...     segment_cols=[['age_band'], ['channel']],
... )

When to use BCF vs. DML
-----------------------
Use BCF (this library) when:
- Treatment is binary or categorical (rate increase yes/no, NCD tier, telematics)
- You want posterior uncertainty for audit documentation
- Strong confounding is suspected (risk model outputs drive both premium and renewal)
- You want FCA-ready segment credible intervals

Use DML (insurance-elasticity) when:
- Treatment is the actual premium level (continuous)
- You have exogenous price variation (A/B test, GIPP shock as instrument)
- You want a single elasticity scalar for the rate optimiser

Dependencies
------------
- stochtree>=0.4.0,<0.5.0 (BCFModel C++ engine by Herren, Hahn, Murray, Carvalho)
- numpy, pandas, scikit-learn, matplotlib, jinja2

stochtree requires a C++ build. On environments without a stochtree wheel,
the library falls back to a mock implementation for testing. Set
INSURANCE_BCF_USE_MOCK=1 to force the mock regardless of stochtree availability.

Modules
-------
model           BayesianCausalForest
elasticity      ElasticityEstimator
audit           BCFAuditReport
simulate        simulate_renewal, SimulationParams
"""

from .audit import BCFAuditReport
from .elasticity import ElasticityEstimator
from .model import (
    BayesianCausalForest,
    ConvergenceWarning,
    GIPPBreakWarning,
    NotFittedError,
    PositivityViolationError,
)
from .simulate import SimulatedDataset, SimulationParams, simulate_renewal

__version__ = "0.1.0"

__all__ = [
    # Core model
    "BayesianCausalForest",
    # Elasticity / segment reporting
    "ElasticityEstimator",
    # Audit / compliance
    "BCFAuditReport",
    # Simulation
    "SimulationParams",
    "SimulatedDataset",
    "simulate_renewal",
    # Exceptions and warnings
    "GIPPBreakWarning",
    "PositivityViolationError",
    "ConvergenceWarning",
    "NotFittedError",
]
