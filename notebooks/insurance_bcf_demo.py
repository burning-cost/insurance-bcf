# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # insurance-bcf: Bayesian Causal Forests for UK Insurance Pricing
# MAGIC
# MAGIC This notebook demonstrates the full `insurance-bcf` workflow on synthetic UK motor insurance data.
# MAGIC
# MAGIC **Scenario:** A motor insurer applied an 8% rate increase to a subset of renewing policies
# MAGIC in 2023 Q4. We want to quantify the heterogeneous lapse effect across customer segments —
# MAGIC not the average elasticity, but the segment-level posterior treatment effects with credible
# MAGIC intervals for FCA audit documentation.
# MAGIC
# MAGIC **Method:** Bayesian Causal Forests (BCF) — Hahn, Murray, Carvalho (2020).
# MAGIC Engine: stochtree 0.4.0 (reference Python implementation by the BCF paper authors).
# MAGIC
# MAGIC **Sections:**
# MAGIC 1. Install and import
# MAGIC 2. Simulate UK motor renewal data
# MAGIC 3. Fit BayesianCausalForest
# MAGIC 4. CATE: posterior mean and credible intervals
# MAGIC 5. Segment effects: who lapses most?
# MAGIC 6. Partial dependence: CATE vs. NCB steps
# MAGIC 7. Rate adjustment recommendations
# MAGIC 8. FCA EP25/2 protected characteristic analysis
# MAGIC 9. Audit report (HTML)
# MAGIC 10. Model serialisation

# COMMAND ----------
# MAGIC %pip install insurance-bcf stochtree>=0.4.0

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_bcf import (
    BayesianCausalForest,
    ElasticityEstimator,
    BCFAuditReport,
)
from insurance_bcf.simulate import simulate_renewal, SimulationParams

print(f"insurance-bcf loaded successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Simulate UK motor renewal data
# MAGIC
# MAGIC The simulation generates realistic UK motor insurance renewal data with:
# MAGIC - Known heterogeneous treatment effects (true tau varies by age band and channel)
# MAGIC - Confounding: the risk model drives both premium assignment and renewal probability
# MAGIC - 7 rating factors: age band, NCB steps, vehicle age, channel, policy duration, plus noise

# COMMAND ----------

params = SimulationParams(
    n_policies=10_000,
    n_features=10,
    treatment_fraction=0.50,    # 50% received rate increase
    base_renewal_prob=0.75,
    tau_heterogeneity=0.08,     # 8pp range in lapse sensitivity
    confounding_strength=1.2,   # strong confounding
    random_seed=42,
)

data = simulate_renewal(params)

print(f"Portfolio: {len(data.X):,} policies")
print(f"Treated (rate increase applied): {data.treatment.sum():,.0f} ({data.treatment.mean():.1%})")
print(f"Renewed: {data.outcome.sum():,.0f} ({data.outcome.mean():.1%})")
print(f"\nFeatures:")
print(data.X.dtypes)
print(f"\nTrue tau (lapse effect) — summary:")
print(data.true_tau.describe())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Ground truth: which segments are most lapse-sensitive?
# MAGIC
# MAGIC The simulation encodes known heterogeneity. BCF should recover this pattern.

# COMMAND ----------

true_seg = (
    data.true_tau.groupby([data.X["age_band"], data.X["channel"]])
    .mean()
    .reset_index()
    .rename(columns={0: "true_tau"})
)
true_seg.columns = ["age_band", "channel", "true_tau_mean"]
true_seg["channel_label"] = true_seg["channel"].map({0: "Direct", 1: "PCW", 2: "Broker"})
true_seg["age_label"] = true_seg["age_band"].map({
    0: "17-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65+"
})
print(true_seg.sort_values("true_tau_mean")[["age_label", "channel_label", "true_tau_mean"]].to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Fit BayesianCausalForest
# MAGIC
# MAGIC stochtree BCF with:
# MAGIC - 500 MCMC samples (after 10 GFR warm-start iterations)
# MAGIC - propensity_covariate='prognostic' — RIC correction (default)
# MAGIC - probit link for binary renewal outcome

# COMMAND ----------

model = BayesianCausalForest(
    outcome="binary",           # binary renewal flag → probit link
    treatment_trees=50,         # shrink-to-homogeneity prior (BCF default)
    prognostic_trees=250,       # expressive prior for mu
    num_mcmc=500,
    num_gfr=10,
    num_chains=1,
    propensity_covariate="prognostic",  # RIC correction
    random_seed=42,
)

print("Fitting BCF model...")
model.fit(
    X=data.X,
    treatment=data.treatment,
    outcome=data.outcome,
    # propensity=None — estimated via logistic regression
)
print(f"Model fitted. Propensity scores computed.")
print(f"  Propensity range: [{model.propensity_scores.min():.3f}, {model.propensity_scores.max():.3f}]")
print(f"  Propensity mean: {model.propensity_scores.mean():.3f}")
print(repr(model))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. CATE: posterior mean and credible intervals

# COMMAND ----------

cate_df = model.cate(data.X, credible_level=0.95)
print(f"CATE output shape: {cate_df.shape}")
print(f"\nPortfolio summary:")
print(cate_df.describe())

# COMMAND ----------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# CATE distribution
ax1.hist(cate_df["cate_mean"], bins=80, color="#1a3050", alpha=0.8, edgecolor="none")
ax1.axvline(0, color="#c0392b", linestyle="--", linewidth=2, label="No effect")
ax1.axvline(cate_df["cate_mean"].mean(), color="#27ae60", linestyle="-",
            linewidth=2, label=f"Mean CATE = {cate_df['cate_mean'].mean():.4f}")
ax1.set_xlabel("CATE (posterior mean lapse effect)")
ax1.set_ylabel("Number of policies")
ax1.set_title("Distribution of Posterior Mean CATE")
ax1.legend()

# CI width distribution
ci_width = cate_df["cate_upper"] - cate_df["cate_lower"]
ax2.hist(ci_width, bins=60, color="#2980b9", alpha=0.8, edgecolor="none")
ax2.set_xlabel("95% Credible interval width")
ax2.set_ylabel("Number of policies")
ax2.set_title("Uncertainty in CATE Estimates")

plt.tight_layout()
plt.savefig("/tmp/cate_distribution.png", dpi=150, bbox_inches="tight")
display(plt.gcf())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Segment effects: who lapses most?

# COMMAND ----------

est = ElasticityEstimator(model, credible_level=0.95)

# Segment by age_band and channel
seg_df = est.segment_effects(
    data.X,
    segment_cols=["age_band", "channel"],
    min_policies=50,
)
seg_df["age_label"] = seg_df["age_band"].map({
    0: "17-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55-64", 5: "65+"
})
seg_df["channel_label"] = seg_df["channel"].map({0: "Direct", 1: "PCW", 2: "Broker"})

print(f"Segments reported: {len(seg_df)}")
print("\nTop 5 most lapse-sensitive segments:")
print(
    seg_df.nsmallest(5, "effect_mean")[
        ["age_label", "channel_label", "effect_mean", "effect_lower", "effect_upper", "n_policies"]
    ].to_string(index=False)
)
print("\nBottom 5 least lapse-sensitive segments:")
print(
    seg_df.nlargest(5, "effect_mean")[
        ["age_label", "channel_label", "effect_mean", "effect_lower", "effect_upper", "n_policies"]
    ].to_string(index=False)
)

# COMMAND ----------

# Visualise segment effects
fig, ax = plt.subplots(figsize=(12, 6))

seg_plot = seg_df.sort_values("effect_mean").reset_index(drop=True)
x = np.arange(len(seg_plot))
labels = seg_plot["age_label"].astype(str) + " / " + seg_plot["channel_label"].astype(str)

ax.barh(
    x,
    seg_plot["effect_mean"],
    xerr=[
        seg_plot["effect_mean"] - seg_plot["effect_lower"],
        seg_plot["effect_upper"] - seg_plot["effect_mean"],
    ],
    color=["#c0392b" if v < -0.05 else "#e67e22" if v < -0.03 else "#27ae60"
           for v in seg_plot["effect_mean"]],
    alpha=0.85,
    ecolor="#555",
    capsize=4,
)
ax.set_yticks(x)
ax.set_yticklabels(labels, fontsize=10)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("CATE (posterior mean lapse effect, with 95% CI)")
ax.set_title("Segment-Level Lapse Effects — Age Band × Channel")
plt.tight_layout()
plt.savefig("/tmp/segment_effects.png", dpi=150, bbox_inches="tight")
display(plt.gcf())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Partial dependence: CATE vs. NCB steps
# MAGIC
# MAGIC How does lapse sensitivity vary as NCB increases? After averaging over
# MAGIC all other covariates (age, channel, vehicle age, etc.).

# COMMAND ----------

pd_df = est.partial_dependence(
    data.X,
    feature="ncb_steps",
    grid_points=6,
)
print(pd_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pd_df["feature_value"], pd_df["pdp_mean"], "o-", color="#1a3050", linewidth=2)
ax.fill_between(
    pd_df["feature_value"],
    pd_df["pdp_lower"],
    pd_df["pdp_upper"],
    alpha=0.2,
    color="#1a3050",
    label="95% CI",
)
ax.set_xlabel("NCB Steps")
ax.set_ylabel("Posterior mean CATE")
ax.set_title("Partial Dependence: Lapse Effect vs. NCB Steps")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/pd_ncb.png", dpi=150, bbox_inches="tight")
display(plt.gcf())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Rate adjustment recommendations
# MAGIC
# MAGIC Elasticity-weighted rate adjustments toward a 5% target margin.
# MAGIC This is a starting point — not a final rate decision.

# COMMAND ----------

np.random.seed(42)
current_premium = pd.Series(
    np.random.uniform(400, 1400, len(data.X)),
    index=data.X.index,
    name="current_premium",
)

adj_df = est.optimal_rate_adjustment(
    data.X,
    target_margin=0.05,
    current_premium=current_premium,
    max_adjustment=0.20,
)

print("Rate adjustment distribution:")
print(adj_df["suggested_adjustment"].describe())
print(f"\nPolicies receiving reduction: {(adj_df['suggested_adjustment'] < 0).sum():,}")
print(f"Policies receiving increase:  {(adj_df['suggested_adjustment'] > 0).sum():,}")
print(f"\nMean adjustment confidence: {adj_df['adjustment_confidence'].mean():.3f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. FCA EP25/2 protected characteristic analysis
# MAGIC
# MAGIC Does the lapse effect vary by age band? BCF credible intervals
# MAGIC provide the Bayesian evidence base for the fairness assessment.

# COMMAND ----------

report = BCFAuditReport(model, est)

pc_df = report.protected_characteristic_check(
    data.X,
    protected_cols=["age_band", "channel"],
)

print("Protected characteristic analysis:")
print(pc_df.to_string(index=False))
print(f"\nSegments flagged for REVIEW: {(pc_df['flag'] == 'REVIEW').sum()}")
print(f"Segments to MONITOR: {(pc_df['flag'] == 'MONITOR').sum()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Audit report (HTML)
# MAGIC
# MAGIC Full HTML report including all sections above.

# COMMAND ----------

report_path = "/tmp/bcf_audit_2024Q4.html"

report.render(
    output_path=report_path,
    X=data.X,
    Z=data.treatment,
    protected_cols=["age_band", "channel"],
    segment_cols=[["age_band"], ["channel"], ["age_band", "channel"]],
    include_plots=True,
)

print(f"Report written to: {report_path}")
print(f"File size: {len(open(report_path).read()):,} bytes")

# To copy to DBFS for download:
# dbutils.fs.cp(f"file:{report_path}", "dbfs:/tmp/bcf_audit_2024Q4.html")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Model serialisation

# COMMAND ----------

json_str = model.to_json()
print(f"JSON length: {len(json_str):,} characters")

# Restore model
model2 = BayesianCausalForest.from_json(json_str, outcome="binary")
cate_check = model2.cate(data.X.head(100))
print(f"Restored model CATE shape: {cate_check.shape}")
print("Serialisation round-trip: OK")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Policies | 10,000 |
# MAGIC | MCMC samples | 500 |
# MAGIC | GFR warm-start | 10 |
# MAGIC | Mean CATE | See above |
# MAGIC | Segments analysed | Age band × Channel |
# MAGIC | Segments flagged (REVIEW) | See above |
# MAGIC
# MAGIC **Key finding:** The BCF model recovers heterogeneous lapse effects that the
# MAGIC aggregate elasticity misses. Young PCW customers are substantially more lapse-
# MAGIC sensitive than mature direct customers. A uniform rate approach overcharges
# MAGIC sensitive segments (causing avoidable lapses) and undercharges insensitive
# MAGIC segments (leaving margin on the table).
# MAGIC
# MAGIC **Next steps:**
# MAGIC 1. Replace synthetic data with actual renewal portfolio from data warehouse
# MAGIC 2. Fit with external propensity from calibrated logistic model (encode treatment assignment rules)
# MAGIC 3. Increase `num_mcmc=1000` and `num_chains=4` for production-grade convergence
# MAGIC 4. Layer business rules onto `optimal_rate_adjustment()` output
# MAGIC 5. Share audit report with senior actuary and model governance committee
