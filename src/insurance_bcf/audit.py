"""
BCFAuditReport — FCA EP25/2 compliance documentation for BCF models.

FCA Evaluation Paper EP25/2 (2025) reviewed the GIPP remedy outcomes and
found material heterogeneity in lapse effects across customer segments. For
ongoing post-GIPP monitoring, insurers must demonstrate that rate changes
have proportionate effects across customer segments and do not create
unfair outcomes for protected characteristic groups.

This module produces an HTML audit report documenting:
1. Model configuration and fit summary
2. MCMC convergence diagnostics
3. Segment-level CATE with credible intervals
4. Protected characteristic moderation analysis
5. Methodology appendix (unconfoundedness assumption, RIC correction rationale)

The report format is designed for internal model governance, not for
submission to the FCA directly. It provides the statistical evidence base
that a quant team would present to a senior actuarial or risk committee.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from jinja2 import Environment, BaseLoader

from .elasticity import ElasticityEstimator
from .model import BayesianCausalForest, NotFittedError


# ------------------------------------------------------------------
# HTML template
# ------------------------------------------------------------------

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BCF Audit Report — {{ report_date }}</title>
<style>
  body { font-family: Arial, sans-serif; font-size: 13px; color: #222; max-width: 1100px; margin: 40px auto; padding: 0 20px; }
  h1 { color: #1a3050; border-bottom: 2px solid #1a3050; padding-bottom: 8px; }
  h2 { color: #1a3050; margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
  h3 { color: #2c5282; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; }
  th { background-color: #1a3050; color: white; padding: 8px 10px; text-align: left; }
  td { padding: 6px 10px; border-bottom: 1px solid #e8e8e8; }
  tr:nth-child(even) { background-color: #f7f9fb; }
  .flag-red { color: #c0392b; font-weight: bold; }
  .flag-amber { color: #e67e22; font-weight: bold; }
  .flag-green { color: #27ae60; }
  .summary-box { background: #f0f4f8; border-left: 4px solid #1a3050; padding: 12px 16px; margin: 16px 0; }
  .warning-box { background: #fff8e1; border-left: 4px solid #e67e22; padding: 12px 16px; margin: 16px 0; }
  .methodology { background: #f8f8f8; border: 1px solid #ddd; padding: 16px; margin-top: 24px; font-size: 12px; }
  img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
  .footer { margin-top: 48px; border-top: 1px solid #ccc; padding-top: 16px; font-size: 11px; color: #666; }
</style>
</head>
<body>

<h1>Bayesian Causal Forest — Audit Report</h1>
<div class="summary-box">
  <strong>Generated:</strong> {{ report_date }}<br>
  <strong>Model:</strong> BayesianCausalForest (insurance-bcf {{ version }})<br>
  <strong>Engine:</strong> stochtree BCFModel (Herren, Hahn, Murray, Carvalho 2025/2026)<br>
  <strong>Outcome type:</strong> {{ outcome }}<br>
  <strong>MCMC samples:</strong> {{ num_mcmc }}<br>
  <strong>Training policies:</strong> {{ "{:,}".format(n_train) }}<br>
  <strong>Features:</strong> {{ n_features }}
</div>

{% if has_warning %}
<div class="warning-box">
  <strong>Note:</strong> {{ warning_text }}
</div>
{% endif %}

<h2>1. Model Configuration</h2>
<table>
<tr><th>Parameter</th><th>Value</th><th>Note</th></tr>
<tr><td>outcome</td><td>{{ outcome }}</td><td>{% if outcome == 'binary' %}Probit link active — tau on latent scale{% else %}Linear link{% endif %}</td></tr>
<tr><td>treatment_trees</td><td>{{ treatment_trees }}</td><td>Shrink-to-homogeneity prior (alpha=0.25, beta=3)</td></tr>
<tr><td>prognostic_trees</td><td>{{ prognostic_trees }}</td><td>Expressive prior (alpha=0.95, beta=2)</td></tr>
<tr><td>num_mcmc</td><td>{{ num_mcmc }}</td><td>Retained posterior samples after GFR warm-start</td></tr>
<tr><td>num_gfr</td><td>{{ num_gfr }}</td><td>GFR warm-start iterations (He &amp; Hahn 2021)</td></tr>
<tr><td>num_chains</td><td>{{ num_chains }}</td><td>{% if num_chains > 1 %}R-hat diagnostics available{% else %}Single chain — no R-hat{% endif %}</td></tr>
<tr><td>propensity_covariate</td><td>{{ propensity_covariate }}</td><td>RIC correction — pi_hat included in mu forest</td></tr>
</table>

<h2>2. Portfolio CATE Summary</h2>
{{ portfolio_summary_table }}

{% if cate_plot %}
<h3>CATE Distribution</h3>
<img src="data:image/png;base64,{{ cate_plot }}" alt="CATE distribution">
{% endif %}

{% if propensity_plot %}
<h3>Propensity Score Distribution</h3>
<img src="data:image/png;base64,{{ propensity_plot }}" alt="Propensity score distribution">
{% endif %}

{% if convergence_table %}
<h2>3. MCMC Convergence Diagnostics</h2>
{{ convergence_table }}
{% endif %}

{% if segment_tables %}
<h2>4. Segment-Level Effects</h2>
{% for seg_html in segment_tables %}
{{ seg_html }}
{% endfor %}
{% endif %}

{% if protected_table %}
<h2>5. Protected Characteristic Analysis</h2>
<p>
  FCA EP25/2 requires demonstration that rate changes have proportionate effects
  across customer segments, including groups defined by protected characteristics.
  The table below shows posterior CATE by protected group. The "Differs from avg"
  column flags groups where the 95% credible interval does not overlap with the
  portfolio average CATE.
</p>
{{ protected_table }}
{% endif %}

<h2>{{ methodology_section_num }}. Methodology</h2>
<div class="methodology">
<h3>Model Equation</h3>
<pre>Y_i = mu(x_i, pi_hat(x_i)) + tau(x_i) * z_i + epsilon_i</pre>
<p>
  <strong>mu(x, pi_hat)</strong>: prognostic function — expected outcome under control.
  pi_hat is included as a covariate to address Regularization-Induced Confounding (RIC).
</p>
<p>
  <strong>tau(x)</strong>: treatment effect function — the CATE (Conditional Average Treatment Effect).
  Uses a shrink-to-homogeneity prior (alpha=0.25, beta=3) that forces tau toward a constant
  unless the data strongly supports heterogeneity.
</p>

<h3>Identification Assumption</h3>
<p>
  Strong ignorability (unconfoundedness): {Y(0), Y(1)} &#8869; Z | X.
  This requires that all confounders — variables affecting both treatment assignment
  and the outcome — are captured in X. For insurance renewal pricing, this is
  defensible when: the risk model is fully observed in X, underwriting rules are
  included, and there is no hidden selection beyond the model covariates.
</p>
<p>
  Sensitivity to unmeasured confounding should be assessed using Rosenbaum bounds
  or similar techniques before relying on these estimates for regulatory reporting.
</p>

<h3>Regularization-Induced Confounding (RIC)</h3>
<p>
  In observational insurance data, E[Y|X] ≈ f(pi(X)) — the renewal probability
  is largely explained by the propensity score. Standard BART over-shrinks mu,
  leaving residual variance correlated with Z. BCF corrects this by including
  pi_hat explicitly in the mu forest (propensity_covariate='prognostic').
  Reference: Hahn, Murray, Carvalho (2020) Bayesian Analysis 15(3): 965-1056.
</p>

<h3>References</h3>
<ol>
  <li>Hahn, P.R., Murray, J.S., Carvalho, C.M. (2020). Bayesian Regression Tree Models for Causal Inference. <em>Bayesian Analysis</em> 15(3): 965-1056.</li>
  <li>Herren, A., Hahn, P.R., Murray, J.S., Carvalho, C.M. (2025/2026). StochTree. arXiv:2512.12051v2.</li>
  <li>Chipman, H.A., George, E.I., McCulloch, R.E. (2010). BART: Bayesian Additive Regression Trees. <em>Annals of Applied Statistics</em> 4(1): 266-298.</li>
  <li>FCA Evaluation Paper EP25/2 (2025). Evaluation of GIPP Remedies. Financial Conduct Authority.</li>
</ol>
</div>

<div class="footer">
  Generated by insurance-bcf (Burning Cost). This report is for internal model governance use.
  It does not constitute FCA submission documentation. Review with a qualified actuary before
  use in pricing decisions.
</div>
</body>
</html>"""


# ------------------------------------------------------------------
# BCFAuditReport
# ------------------------------------------------------------------


class BCFAuditReport:
    """
    FCA EP25/2 format audit report for a fitted BayesianCausalForest.

    Parameters
    ----------
    model : BayesianCausalForest
        Fitted model.
    estimator : ElasticityEstimator or None
        If provided, segment-level effects are included in the report.
    """

    def __init__(
        self,
        model: BayesianCausalForest,
        estimator: ElasticityEstimator | None = None,
    ) -> None:
        self.model = model
        self.estimator = estimator

    # ------------------------------------------------------------------
    # protected_characteristic_check
    # ------------------------------------------------------------------

    def protected_characteristic_check(
        self,
        X: pd.DataFrame,
        protected_cols: Sequence[str],
        treatment_value: float = 1.0,
        credible_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Posterior CATE by protected characteristic group.

        Flags groups where the 95% credible interval does not overlap with
        the portfolio average CATE. This is the FCA EP25/2 relevant test:
        are rate change effects proportionate across protected groups?

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        protected_cols : sequence of str
            Column names for protected characteristics. Each column is
            analysed separately (not jointly), following the FCA approach
            of characteristic-by-characteristic analysis.
        treatment_value : float
            Treatment level.
        credible_level : float
            Credible interval level. Default 0.95.

        Returns
        -------
        pd.DataFrame
            Columns: characteristic, group, effect_mean, effect_lower,
                     effect_upper, n_policies, differs_from_avg, flag.
        """
        if not self.model._is_fitted:
            raise NotFittedError("Model is not fitted.")

        alpha = (1.0 - credible_level) / 2.0

        # Portfolio average CATE
        tau_post_all = self.model.posterior_samples(X, treatment_value=treatment_value)
        portfolio_mean_per_sample = tau_post_all.mean(axis=0)
        portfolio_mean = float(portfolio_mean_per_sample.mean())
        portfolio_lower = float(np.quantile(portfolio_mean_per_sample, alpha))
        portfolio_upper = float(np.quantile(portfolio_mean_per_sample, 1.0 - alpha))

        records = []

        for col in protected_cols:
            if col not in X.columns:
                warnings.warn(
                    f"Protected column {col!r} not in X. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            unique_groups = X[col].dropna().unique()

            for group_val in sorted(unique_groups, key=str):
                mask = X[col] == group_val
                n = int(mask.sum())
                if n == 0:
                    continue

                # Use .iloc to re-index correctly
                group_idx = mask[mask].index
                position_idx = [X.index.get_loc(i) for i in group_idx]
                tau_group = tau_post_all[position_idx].mean(axis=0)

                g_mean = float(tau_group.mean())
                g_lower = float(np.quantile(tau_group, alpha))
                g_upper = float(np.quantile(tau_group, 1.0 - alpha))

                # Differs from avg: group CI does not overlap with portfolio avg
                # (simple non-overlap test between group CI and portfolio CI)
                differs = bool(
                    g_upper < portfolio_lower or g_lower > portfolio_upper
                )

                if differs:
                    flag = "REVIEW"
                elif abs(g_mean - portfolio_mean) > 0.5 * (portfolio_upper - portfolio_lower):
                    flag = "MONITOR"
                else:
                    flag = "OK"

                records.append(
                    {
                        "characteristic": col,
                        "group": str(group_val),
                        "effect_mean": g_mean,
                        "effect_lower": g_lower,
                        "effect_upper": g_upper,
                        "n_policies": n,
                        "differs_from_avg": differs,
                        "flag": flag,
                    }
                )

        # Append portfolio average row
        records.append(
            {
                "characteristic": "PORTFOLIO",
                "group": "all",
                "effect_mean": portfolio_mean,
                "effect_lower": portfolio_lower,
                "effect_upper": portfolio_upper,
                "n_policies": len(X),
                "differs_from_avg": False,
                "flag": "BASELINE",
            }
        )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(
        self,
        output_path: str,
        X: pd.DataFrame | None = None,
        Z: pd.Series | None = None,
        y: pd.Series | None = None,
        protected_cols: Sequence[str] | None = None,
        segment_cols: Sequence[Sequence[str]] | None = None,
        treatment_value: float = 1.0,
        include_plots: bool = True,
    ) -> None:
        """
        Render the audit report to an HTML file.

        Parameters
        ----------
        output_path : str
            Path to write the HTML file.
        X : pd.DataFrame or None
            Covariate matrix for CATE computation. If None, report contains
            configuration only.
        Z : pd.Series or None
            Treatment series (used for propensity plot if X provided).
        y : pd.Series or None
            Outcome series (currently unused in report, reserved).
        protected_cols : sequence of str or None
            Protected characteristic columns for Section 5.
        segment_cols : sequence of sequences or None
            List of segment column groupings for Section 4.
            e.g. [['age_band'], ['age_band', 'channel']].
        treatment_value : float
            Treatment level for CATE computation.
        include_plots : bool
            Whether to embed matplotlib plots. True by default.
        """
        if not self.model._is_fitted:
            raise NotFittedError("Model is not fitted. Call fit() before rendering.")

        import base64
        from datetime import date

        report_date = date.today().isoformat()

        context: dict = {
            "report_date": report_date,
            "version": _get_version(),
            "outcome": self.model.outcome,
            "num_mcmc": self.model.num_mcmc,
            "num_gfr": self.model.num_gfr,
            "num_chains": self.model.num_chains,
            "treatment_trees": self.model.treatment_trees,
            "prognostic_trees": self.model.prognostic_trees,
            "propensity_covariate": self.model.propensity_covariate,
            "n_train": self.model._n_train,
            "n_features": len(self.model._feature_names),
            "has_warning": False,
            "warning_text": "",
            "portfolio_summary_table": "",
            "cate_plot": "",
            "propensity_plot": "",
            "convergence_table": "",
            "segment_tables": [],
            "protected_table": "",
            "methodology_section_num": 2,
        }

        section_num = 2

        if X is not None:
            # Portfolio summary
            if self.estimator is not None:
                try:
                    summary = self.estimator.portfolio_summary(X, treatment_value=treatment_value)
                    context["portfolio_summary_table"] = _df_to_html(summary)
                    section_num += 1
                except Exception as e:
                    context["has_warning"] = True
                    context["warning_text"] = f"Portfolio summary failed: {e}"

            # CATE plot
            if include_plots:
                try:
                    cate_df = self.model.cate(X, treatment_value=treatment_value)
                    cate_b64 = _cate_distribution_plot(cate_df["cate_mean"].to_numpy())
                    context["cate_plot"] = cate_b64
                except Exception:
                    pass

            # Propensity plot
            if include_plots and self.model.propensity_scores is not None and Z is not None:
                try:
                    pi = self.model.propensity_scores
                    Z_arr = np.asarray(Z, dtype=float)
                    prop_b64 = _propensity_plot(pi, Z_arr)
                    context["propensity_plot"] = prop_b64
                except Exception:
                    pass

            # Convergence
            conv_df = self.model.convergence_summary()
            if not conv_df.empty:
                context["convergence_table"] = _df_to_html(conv_df)
                section_num += 1

            # Segment effects
            if segment_cols is not None and self.estimator is not None:
                seg_htmls = []
                for sc in segment_cols:
                    try:
                        seg_df = self.estimator.segment_effects(X, list(sc), treatment_value=treatment_value)
                        label = " x ".join(sc)
                        seg_htmls.append(f"<h3>Segment: {label}</h3>" + _df_to_html(seg_df))
                    except Exception as exc:
                        seg_htmls.append(f"<p>Segment {sc} failed: {exc}</p>")
                context["segment_tables"] = seg_htmls
                section_num += 1

            # Protected characteristic check
            if protected_cols is not None:
                try:
                    pc_df = self.protected_characteristic_check(
                        X, protected_cols, treatment_value=treatment_value
                    )
                    context["protected_table"] = _df_to_html(
                        pc_df,
                        flag_col="flag",
                        flag_map={"REVIEW": "flag-red", "MONITOR": "flag-amber", "OK": "flag-green"},
                    )
                    section_num += 1
                except Exception as exc:
                    context["has_warning"] = True
                    context["warning_text"] = (
                        (context["warning_text"] + " | " if context["warning_text"] else "")
                        + f"Protected characteristic check failed: {exc}"
                    )

        context["methodology_section_num"] = section_num + 1

        env = Environment(loader=BaseLoader())
        template = env.from_string(_REPORT_TEMPLATE)
        html = template.render(**context)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BCFAuditReport(model={self.model!r})"


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------


def _cate_distribution_plot(cate_mean: npt.NDArray[np.float64]) -> str:
    """Histogram of CATE point estimates; returns base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cate_mean, bins=50, color="#1a3050", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="#c0392b", linestyle="--", linewidth=1.5, label="No effect")
    ax.axvline(float(np.mean(cate_mean)), color="#27ae60", linestyle="-",
               linewidth=1.5, label=f"Mean CATE = {np.mean(cate_mean):.4f}")
    ax.set_xlabel("CATE (treatment effect)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Posterior Mean CATE")
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def _propensity_plot(pi: npt.NDArray[np.float64], Z: npt.NDArray[np.float64]) -> str:
    """Propensity score overlap plot; returns base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pi[Z == 0], bins=40, alpha=0.6, color="#1a3050", label="Control (Z=0)", density=True)
    ax.hist(pi[Z == 1], bins=40, alpha=0.6, color="#e67e22", label="Treated (Z=1)", density=True)
    ax.axvline(0.05, color="#c0392b", linestyle="--", linewidth=1, label="Positivity threshold")
    ax.axvline(0.95, color="#c0392b", linestyle="--", linewidth=1)
    ax.set_xlabel("Propensity score P(Z=1 | X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap")
    ax.legend()
    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig: plt.Figure) -> str:
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _df_to_html(
    df: pd.DataFrame,
    flag_col: str | None = None,
    flag_map: dict | None = None,
) -> str:
    """Convert DataFrame to HTML table string with optional flag styling."""
    if df.empty:
        return "<p><em>No data available.</em></p>"

    def _fmt(val: object) -> str:
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    rows_html = []
    for _, row in df.iterrows():
        cells = []
        flag_class = ""
        if flag_col and flag_col in row and flag_map:
            flag_class = flag_map.get(str(row[flag_col]), "")

        for col in df.columns:
            val = row[col]
            cell_class = ""
            if col == flag_col and flag_map:
                cell_class = flag_map.get(str(val), "")
            elif flag_class:
                cell_class = flag_class

            cell = f'<td class="{cell_class}">{_fmt(val)}</td>' if cell_class else f"<td>{_fmt(val)}</td>"
            cells.append(cell)
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    table = f"<table><tr>{headers}</tr>{''.join(rows_html)}</table>"
    return table


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("insurance-bcf")
    except Exception:
        return "dev"
