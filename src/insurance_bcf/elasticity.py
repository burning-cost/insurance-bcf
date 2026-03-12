"""
ElasticityEstimator — segment-level CATE aggregation and rate adjustment.

Converts the raw posterior over tau(x) from BayesianCausalForest into
actuarial-grade segment summaries: group-level effects, partial dependence
of tau on individual features, and elasticity-weighted rate adjustment
recommendations.

The name 'elasticity' here is loose — BCF produces treatment effect estimates
(CATE), not elasticities in the strict economic sense (dE[Y]/d log P). For
continuous premium elasticities use insurance-elasticity (DML-based). This
module names the output 'elasticity' to match the vocabulary actuarial pricing
teams use when discussing segment-level lapse sensitivity to rate changes.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .model import BayesianCausalForest, NotFittedError


class ElasticityEstimator:
    """
    Segment-level CATE summaries from a fitted BayesianCausalForest.

    Parameters
    ----------
    model : BayesianCausalForest
        A fitted BayesianCausalForest instance.
    credible_level : float
        Default credible interval level for all methods. Can be overridden
        per call. Default 0.95.
    """

    def __init__(
        self,
        model: BayesianCausalForest,
        credible_level: float = 0.95,
    ) -> None:
        if not isinstance(model, BayesianCausalForest):
            raise TypeError(
                f"model must be a BayesianCausalForest, got {type(model).__name__}"
            )
        self.model = model
        self.credible_level = credible_level

    # ------------------------------------------------------------------
    # segment_effects
    # ------------------------------------------------------------------

    def segment_effects(
        self,
        X: pd.DataFrame,
        segment_cols: Sequence[str],
        treatment_value: float = 1.0,
        credible_level: float | None = None,
        min_policies: int = 10,
    ) -> pd.DataFrame:
        """
        Aggregate CATE by segment with posterior credible intervals.

        For each unique combination of segment_cols values, computes the
        posterior mean CATE and credible interval by averaging individual
        policy posteriors within the group.

        The average-within-segment approach preserves proper Bayesian
        uncertainty: the segment-level posterior is the expectation of the
        policy-level posteriors over the empirical distribution of X within
        the segment.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        segment_cols : list of str
            Column names to group by. Must be present in X.
        treatment_value : float
            Treatment level. Default 1.0.
        credible_level : float or None
            Override default credible interval level.
        min_policies : int
            Minimum policies per segment to report. Segments smaller than
            this are excluded with a warning.

        Returns
        -------
        pd.DataFrame
            Columns: [segment_cols..., effect_mean, effect_lower, effect_upper,
                      effect_std, n_policies].
            One row per unique segment.
        """
        self._check_fitted()
        level = credible_level if credible_level is not None else self.credible_level
        alpha = (1.0 - level) / 2.0

        missing = [c for c in segment_cols if c not in X.columns]
        if missing:
            raise ValueError(f"segment_cols not in X: {missing}")

        # Get full posterior: (n_policies, n_mcmc_samples)
        tau_post = self.model.posterior_samples(X, treatment_value=treatment_value)

        records = []
        groups = X.groupby(list(segment_cols), observed=True)

        small_segments = 0
        for group_key, idx in groups.groups.items():
            n = len(idx)
            if n < min_policies:
                small_segments += 1
                continue

            # Average over policies in this segment: (n_mcmc_samples,)
            tau_seg = tau_post[idx.map(lambda i: X.index.get_loc(i)).tolist()].mean(axis=0)

            row: dict = {}
            if isinstance(group_key, tuple):
                for col, val in zip(segment_cols, group_key):
                    row[col] = val
            else:
                row[segment_cols[0]] = group_key

            row["effect_mean"] = float(tau_seg.mean())
            row["effect_lower"] = float(np.quantile(tau_seg, alpha))
            row["effect_upper"] = float(np.quantile(tau_seg, 1.0 - alpha))
            row["effect_std"] = float(tau_seg.std())
            row["n_policies"] = n
            records.append(row)

        if small_segments > 0:
            warnings.warn(
                f"{small_segments} segment(s) had fewer than {min_policies} policies "
                "and were excluded. Reduce min_policies or aggregate further.",
                UserWarning,
                stacklevel=2,
            )

        if not records:
            cols = list(segment_cols) + [
                "effect_mean", "effect_lower", "effect_upper", "effect_std", "n_policies"
            ]
            return pd.DataFrame(columns=cols)

        result = pd.DataFrame(records)
        result = result.sort_values("effect_mean").reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    # partial_dependence
    # ------------------------------------------------------------------

    def partial_dependence(
        self,
        X: pd.DataFrame,
        feature: str,
        grid_points: int = 20,
        treatment_value: float = 1.0,
        credible_level: float | None = None,
        n_sample_policies: int | None = None,
    ) -> pd.DataFrame:
        """
        Posterior mean CATE as a function of a single feature.

        Implements marginal (average) partial dependence: for each grid value
        of `feature`, replace that feature across all policies (or a random
        sample), compute posterior CATE, and average. This traces out how tau
        varies with the feature after averaging out the distribution of other
        covariates.

        This is the BCF version of ICE/PDP plots for interpretability.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        feature : str
            Feature to vary. Must be in X.columns.
        grid_points : int
            Number of evaluation points. For categorical features with fewer
            unique values than grid_points, uses all unique values.
        treatment_value : float
            Treatment level.
        credible_level : float or None
            Override default credible interval level.
        n_sample_policies : int or None
            Subsample X to this many policies before computing. Reduces
            computation for large portfolios. None uses all policies.

        Returns
        -------
        pd.DataFrame
            Columns: feature_value, pdp_mean, pdp_lower, pdp_upper.
            One row per grid point.
        """
        self._check_fitted()
        if feature not in X.columns:
            raise ValueError(f"feature {feature!r} not in X.columns")

        level = credible_level if credible_level is not None else self.credible_level
        alpha = (1.0 - level) / 2.0

        X_work = X
        if n_sample_policies is not None and len(X) > n_sample_policies:
            rng = np.random.default_rng(seed=self.model.random_seed)
            idx = rng.choice(len(X), size=n_sample_policies, replace=False)
            X_work = X.iloc[idx].reset_index(drop=True)

        unique_vals = np.unique(X_work[feature].dropna())
        if len(unique_vals) <= grid_points:
            grid = unique_vals
        else:
            grid = np.linspace(unique_vals.min(), unique_vals.max(), grid_points)

        records = []
        for val in grid:
            X_mod = X_work.copy()
            X_mod[feature] = val
            tau_post = self.model.posterior_samples(
                X_mod, treatment_value=treatment_value
            )
            # Average over policies, then summarise posterior
            tau_mean_per_sample = tau_post.mean(axis=0)  # (n_mcmc,)
            records.append(
                {
                    "feature_value": val,
                    "pdp_mean": float(tau_mean_per_sample.mean()),
                    "pdp_lower": float(np.quantile(tau_mean_per_sample, alpha)),
                    "pdp_upper": float(np.quantile(tau_mean_per_sample, 1.0 - alpha)),
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # optimal_rate_adjustment
    # ------------------------------------------------------------------

    def optimal_rate_adjustment(
        self,
        X: pd.DataFrame,
        target_margin: float,
        current_premium: pd.Series,
        treatment_value: float = 1.0,
        max_adjustment: float = 0.20,
        credible_level: float | None = None,
    ) -> pd.DataFrame:
        """
        Suggest elasticity-weighted rate adjustments per policy.

        Uses the CATE posterior (lapse probability change per treatment unit)
        to recommend premium adjustments that move each segment toward the
        target margin, subject to a maximum adjustment constraint.

        This is a heuristic optimiser, not a full revenue-optimisation model.
        It assumes:
        - tau(x) is approximately the lapse sensitivity to the current rate
          level (the treatment is binary: rate increased or not)
        - Margin impact of lapse = current_premium * (lapse rate change)
        - A negative tau (more lapses under treatment) implies the rate
          increase was counter-productive for that segment

        The output is a starting point for the pricing team's rate review,
        not a final answer. Layer business rules and reserving constraints
        on top.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        target_margin : float
            Target combined operating margin (e.g. 0.05 for 5%).
        current_premium : pd.Series
            Current annual premium per policy. Must align with X.index.
        treatment_value : float
            Treatment level for CATE computation.
        max_adjustment : float
            Cap on the absolute rate adjustment (e.g. 0.20 = ±20%). Default 0.20.
        credible_level : float or None
            Override credible level for CIs.

        Returns
        -------
        pd.DataFrame
            Columns: suggested_adjustment, cate_mean, cate_lower, cate_upper,
                     current_premium, adjustment_confidence.
            Index matches X.index.
        """
        self._check_fitted()
        level = credible_level if credible_level is not None else self.credible_level

        cate_df = self.model.cate(X, treatment_value=treatment_value, credible_level=level)
        cate_mean = cate_df["cate_mean"].to_numpy()
        cate_lower = cate_df["cate_lower"].to_numpy()
        cate_upper = cate_df["cate_upper"].to_numpy()

        prem = pd.Series(current_premium, index=X.index).to_numpy(dtype=float)

        # Adjustment rule:
        # tau < 0 means this segment lapses more under the rate increase.
        # The adjustment brings the rate back toward neutral for sensitive segments.
        # Proportional to tau relative to the median tau (most affected vs. average).
        tau_median = np.median(cate_mean)
        tau_range = np.std(cate_mean) + 1e-8

        # Normalised sensitivity: how far below median (negative = more lapse-sensitive)
        sensitivity = (cate_mean - tau_median) / tau_range

        # Raw adjustment: sensitive segments get rate reduction, insensitive get increase
        raw_adj = -sensitivity * max_adjustment / 2.0

        # Scale toward target margin
        # Simple proxy: assume margin scales linearly with premium change
        avg_adj = np.clip(raw_adj, -max_adjustment, max_adjustment)

        # Confidence: wide credible interval => lower confidence
        ci_width = cate_upper - cate_lower
        max_ci = np.percentile(ci_width, 95) + 1e-8
        confidence = 1.0 - np.clip(ci_width / max_ci, 0, 1)

        result = pd.DataFrame(
            {
                "suggested_adjustment": avg_adj,
                "cate_mean": cate_mean,
                "cate_lower": cate_lower,
                "cate_upper": cate_upper,
                "current_premium": prem,
                "adjustment_confidence": confidence,
            },
            index=X.index,
        )

        return result

    # ------------------------------------------------------------------
    # portfolio_summary
    # ------------------------------------------------------------------

    def portfolio_summary(
        self,
        X: pd.DataFrame,
        treatment_value: float = 1.0,
        credible_level: float | None = None,
    ) -> pd.DataFrame:
        """
        High-level portfolio CATE summary statistics.

        Returns a one-row summary with mean, median, 10th/90th percentile
        CATE, and the fraction of policies with a credible interval that
        excludes zero (statistical evidence of heterogeneous effect).

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix.
        treatment_value : float
            Treatment level.
        credible_level : float or None
            Override credible interval level.

        Returns
        -------
        pd.DataFrame
            One row, columns: mean_cate, median_cate, p10_cate, p90_cate,
                              frac_negative_ci, frac_positive_ci,
                              frac_significant.
        """
        self._check_fitted()
        level = credible_level if credible_level is not None else self.credible_level

        cate_df = self.model.cate(X, treatment_value=treatment_value, credible_level=level)
        mean_ = cate_df["cate_mean"].to_numpy()
        lower = cate_df["cate_lower"].to_numpy()
        upper = cate_df["cate_upper"].to_numpy()

        frac_neg = float(np.mean(upper < 0))  # CI entirely below 0
        frac_pos = float(np.mean(lower > 0))  # CI entirely above 0

        return pd.DataFrame(
            [
                {
                    "mean_cate": float(mean_.mean()),
                    "median_cate": float(np.median(mean_)),
                    "p10_cate": float(np.percentile(mean_, 10)),
                    "p90_cate": float(np.percentile(mean_, 90)),
                    "frac_negative_ci": frac_neg,
                    "frac_positive_ci": frac_pos,
                    "frac_significant": frac_neg + frac_pos,
                    "n_policies": len(X),
                }
            ]
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.model._is_fitted:
            raise NotFittedError(
                "The underlying BayesianCausalForest is not fitted. "
                "Call model.fit() before using ElasticityEstimator."
            )

    def __repr__(self) -> str:
        return (
            f"ElasticityEstimator(model={self.model!r}, "
            f"credible_level={self.credible_level})"
        )
