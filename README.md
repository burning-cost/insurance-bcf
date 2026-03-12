# insurance-bcf

Bayesian Causal Forests for UK insurance pricing teams.

## The problem

A motor insurer applies an 8% rate increase across the book. Aggregate lapse rises 1.8pp. The GLM says the elasticity is −0.22. Job done?

No. That number is the average. Young PCW customers may lapse at 3x the rate of mature direct customers under the same rate increase. If you price to the average elasticity, you overshoot on the sensitive segments and leave margin on the insensitive ones.

BCF (Bayesian Causal Forests) estimates the treatment effect for every policy in the portfolio — not an average. The output is a posterior distribution over the lapse effect for each segment, with credible intervals suitable for FCA audit documentation.

## When to use this vs. insurance-elasticity

**Use BCF (this library) when:**
- Treatment is binary or categorical: rate increase applied yes/no, NCD tier change, telematics policy
- You want posterior uncertainty over segment effects for FCA EP25/2 audit documentation
- Strong confounding is suspected: the risk model drives both the premium and the renewal probability
- You want counterfactual analysis: what would have happened if we had not applied the increase to this segment?

**Use DML (insurance-elasticity) when:**
- Treatment is the actual premium level (continuous)
- You have exogenous price variation from an A/B test or natural experiment
- You want a single elasticity scalar to feed into a rate optimiser

The methods are complementary. Run both. Divergence in segment rankings flags model misspecification in one or both.

## The method

BCF runs two separate Bayesian tree ensembles:

```
Y_i = mu(x_i, pi_hat(x_i)) + tau(x_i) * z_i + epsilon_i
```

`mu` — the prognostic function — captures the renewal probability under control (250 trees, expressive prior). `tau` — the treatment effect function — captures CATE (50 trees, shrink-to-homogeneity prior with alpha=0.25, beta=3).

Including `pi_hat` explicitly in `mu` corrects **Regularization-Induced Confounding** (RIC): the mechanism by which standard BART over-shrinks `mu` and incorrectly attributes unexplained outcome variance to the treatment. This is not optional for insurance observational data where the risk model drives both premium assignment and renewal probability.

Reference: Hahn, Murray, Carvalho (2020) *Bayesian Analysis* 15(3): 965-1056.

Engine: [stochtree](https://github.com/StochasticTree/stochtree) 0.4.0 — the reference Python BCF implementation by the original paper authors (Herren, Hahn, Murray, Carvalho 2025/2026).

## Quick start

```python
from insurance_bcf import BayesianCausalForest, ElasticityEstimator, BCFAuditReport
from insurance_bcf.simulate import simulate_renewal, SimulationParams

# Simulate a UK motor renewal dataset (or load your own)
data = simulate_renewal(SimulationParams(n_policies=10_000, random_seed=42))

# Fit the BCF model
model = BayesianCausalForest(
    outcome='binary',       # binary renewal flag
    num_mcmc=500,           # posterior samples
    num_gfr=10,             # GFR warm-start iterations
    random_seed=42,
)
model.fit(
    X=data.X,               # pd.DataFrame of rating factors
    treatment=data.treatment,  # binary: rate increase applied (1) or not (0)
    outcome=data.outcome,   # renewal flag (0/1)
)

# CATE: posterior mean + 95% credible interval per policy
cate_df = model.cate(data.X)
print(cate_df.head())
#    cate_mean  cate_lower  cate_upper  cate_std
# 0   -0.0612     -0.0741     -0.0483    0.0066
# 1   -0.0421     -0.0510     -0.0332    0.0045
# ...

# Segment effects
est = ElasticityEstimator(model)
seg = est.segment_effects(data.X, segment_cols=['age_band', 'channel'])
print(seg)
#   age_band  channel  effect_mean  effect_lower  effect_upper  n_policies
# 0        0        1       -0.082        -0.094        -0.071        1241
# 1        0        0       -0.041        -0.049        -0.033         420
# 2        1        1       -0.035        -0.041        -0.029        3410
# 3        5        0       -0.011        -0.018        -0.004        1892
```

Young PCW customers (age_band=0, channel=1) are 7.5x more lapse-sensitive than mature direct customers. That is the heterogeneity the GLM missed.

## Rate adjustment recommendations

```python
import pandas as pd
import numpy as np

current_premium = pd.Series(np.random.uniform(400, 1200, len(data.X)))

adj = est.optimal_rate_adjustment(
    data.X,
    target_margin=0.05,
    current_premium=current_premium,
    max_adjustment=0.20,
)
print(adj[['suggested_adjustment', 'adjustment_confidence']].head())
```

## Partial dependence

How does the CATE vary with a single feature, after averaging over the distribution of other covariates?

```python
pd_df = est.partial_dependence(data.X, feature='ncb_steps', grid_points=6)
print(pd_df)
# feature_value  pdp_mean  pdp_lower  pdp_upper
#             0    -0.071     -0.082     -0.060
#             1    -0.065     -0.074     -0.055
#             5    -0.031     -0.039     -0.023
```

Customers with higher NCB are less lapse-sensitive to rate increases — they have more to lose by switching insurer.

## FCA EP25/2 audit report

```python
report = BCFAuditReport(model, est)

# Protected characteristic check: does tau vary by age band?
pc_df = report.protected_characteristic_check(
    data.X,
    protected_cols=['age_band'],
)
print(pc_df[['characteristic', 'group', 'effect_mean', 'flag']])

# Render HTML report
report.render(
    output_path='bcf_audit_2024Q4.html',
    X=data.X,
    Z=data.treatment,
    protected_cols=['age_band'],
    segment_cols=[['age_band'], ['channel'], ['age_band', 'channel']],
)
```

The report documents model configuration, MCMC convergence, segment effects, protected characteristic moderation, and a methodology appendix. It is designed for internal model governance, not FCA submission.

## Using pre-computed propensity scores

For insurance applications, passing an external propensity score is preferred over letting BCF estimate it internally. You have domain knowledge about what drives treatment assignment.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(data.X, data.treatment)
pi_hat = lr.predict_proba(data.X)[:, 1]

model.fit(
    X=data.X,
    treatment=data.treatment,
    outcome=data.outcome,
    propensity=pi_hat,
)
```

## GIPP date warning

If your dataset spans January 2022 (the FCA GIPP implementation date), BCF will warn you:

```python
import pandas as pd

X_with_dates = data.X.copy()
X_with_dates['renewal_date'] = pd.date_range('2021-06-01', periods=len(data.X), freq='D')

model.fit(
    X_with_dates, data.treatment, data.outcome,
    gipp_date_col='renewal_date'
)
# GIPPBreakWarning: Column 'renewal_date' spans the GIPP implementation date (January 2022).
```

## Serialisation

```python
# Save
json_str = model.to_json()

# Load
model2 = BayesianCausalForest.from_json(json_str, outcome='binary')
cate_df = model2.cate(data.X)
```

## API reference

### `BayesianCausalForest`

| Parameter | Default | Notes |
|-----------|---------|-------|
| `outcome` | `'binary'` | `'binary'` activates probit link (tau on latent scale); `'continuous'` for loss ratio |
| `treatment_trees` | 50 | Shrink-to-homogeneity prior — do not increase without testing |
| `prognostic_trees` | 250 | Expressive prior for mu |
| `num_mcmc` | 500 | Retained posterior samples |
| `num_gfr` | 10 | GFR warm-start iterations (eliminates burn-in) |
| `num_chains` | 1 | Set to 4 for R-hat diagnostics (requires arviz) |
| `propensity_covariate` | `'prognostic'` | Never `'none'` for observational data |
| `random_seed` | `None` | |
| `positivity_threshold` | 0.05 | Propensity scores outside [0.05, 0.95] |
| `positivity_max_fraction` | 0.05 | Fraction allowed to violate before error |

### `ElasticityEstimator`

| Method | Returns | Notes |
|--------|---------|-------|
| `segment_effects(X, segment_cols)` | `pd.DataFrame` | CATE aggregated by segment |
| `partial_dependence(X, feature)` | `pd.DataFrame` | CATE vs. single feature |
| `optimal_rate_adjustment(X, target_margin, current_premium)` | `pd.DataFrame` | Elasticity-weighted adjustments |
| `portfolio_summary(X)` | `pd.DataFrame` | Aggregate CATE statistics |

### `BCFAuditReport`

| Method | Returns | Notes |
|--------|---------|-------|
| `protected_characteristic_check(X, protected_cols)` | `pd.DataFrame` | FCA EP25/2 protected group analysis |
| `render(output_path, X, ...)` | None | HTML report |

## Installation

```bash
pip install insurance-bcf
```

stochtree requires a C++ build. Wheels are available for Linux x86_64, macOS (Intel + Apple Silicon), and Windows x86_64. On other architectures, the library falls back to a mock implementation for testing.

```bash
pip install stochtree>=0.4.0  # C++ backend
pip install insurance-bcf
```

For MCMC convergence diagnostics with multi-chain sampling:

```bash
pip install insurance-bcf[diagnostics]  # includes arviz
```

## Databricks demo

See `notebooks/insurance_bcf_demo.py` for the full workflow on synthetic data. Upload to your Databricks workspace and run on any cluster with ML runtime >= 13.0.

## Methodology note on binary outcomes

When `outcome='binary'`, BCF uses a probit link function. The treatment effect `tau(x)` is on the **latent normal scale**, not the probability scale. The relationship between latent-scale tau and probability-scale lapse effect depends on mu(x):

```
P(Y=1 | X, Z=1) - P(Y=1 | X, Z=0) = Phi(mu(x) + tau(x)) - Phi(mu(x))
```

For audit reporting, use `posterior_samples(X, marginalise_probit=True)` to apply the standard normal CDF approximation. For precise marginalisation, use `mu` and `tau` posteriors jointly.

## References

1. Hahn, P.R., Murray, J.S., Carvalho, C.M. (2020). Bayesian Regression Tree Models for Causal Inference. *Bayesian Analysis* 15(3): 965-1056.
2. Herren, A., Hahn, P.R., Murray, J.S., Carvalho, C.M. (2025/2026). StochTree. arXiv:2512.12051v2.
3. Chipman, H.A., George, E.I., McCulloch, R.E. (2010). BART. *Annals of Applied Statistics* 4(1): 266-298.
4. He, J., Hahn, P.R. (2021). GFR warm-start algorithm for BART MCMC.
5. FCA Evaluation Paper EP25/2 (2025). Evaluation of GIPP Remedies.

---

*Built by [Burning Cost](https://burning-cost.github.io). Practitioner tools for UK insurance pricing teams.*
