"""Tests for BCFAuditReport."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from insurance_bcf import BayesianCausalForest, BCFAuditReport, ElasticityEstimator, NotFittedError
from insurance_bcf.simulate import SimulationParams, simulate_renewal


@pytest.fixture(scope="module")
def report_fixtures():
    data = simulate_renewal(SimulationParams(n_policies=200, random_seed=20))
    pi = data.true_propensity.to_numpy()
    model = BayesianCausalForest(num_mcmc=30, num_gfr=3, random_seed=0, positivity_max_fraction=0.30)
    model.fit(data.X, data.treatment, data.outcome, propensity=pi)
    est = ElasticityEstimator(model)
    report = BCFAuditReport(model, est)
    return report, model, est, data


class TestBCFAuditReportConstructor:
    def test_accepts_model_only(self, fitted_model):
        report = BCFAuditReport(fitted_model)
        assert report.model is fitted_model
        assert report.estimator is None

    def test_accepts_model_and_estimator(self, fitted_model, fitted_estimator):
        report = BCFAuditReport(fitted_model, fitted_estimator)
        assert report.estimator is fitted_estimator

    def test_repr(self, fitted_model):
        report = BCFAuditReport(fitted_model)
        assert "BCFAuditReport" in repr(report)


class TestProtectedCharacteristicCheck:
    def test_returns_dataframe(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band"]
        )
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band"]
        )
        for col in ["characteristic", "group", "effect_mean", "effect_lower",
                    "effect_upper", "n_policies", "differs_from_avg", "flag"]:
            assert col in result.columns

    def test_includes_portfolio_row(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band"]
        )
        assert "PORTFOLIO" in result["characteristic"].values

    def test_flag_values(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band"]
        )
        valid_flags = {"OK", "MONITOR", "REVIEW", "BASELINE"}
        assert set(result["flag"].unique()).issubset(valid_flags)

    def test_multiple_protected_cols(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band", "channel"]
        )
        chars = result["characteristic"].unique()
        assert "age_band" in chars
        assert "channel" in chars

    def test_missing_col_warns(self, report_fixtures):
        report, model, est, data = report_fixtures
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = report.protected_characteristic_check(
                data.X, protected_cols=["nonexistent_col"]
            )
        warn_msgs = [str(x.message) for x in w]
        assert any("nonexistent_col" in m for m in warn_msgs)

    def test_n_policies_covers_all(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["channel"]
        )
        # Exclude portfolio row; group rows should sum to total
        group_rows = result[result["characteristic"] != "PORTFOLIO"]
        assert group_rows["n_policies"].sum() == len(data.X)

    def test_unfitted_model_raises(self):
        model = BayesianCausalForest()
        report = BCFAuditReport(model)
        data = simulate_renewal(SimulationParams(n_policies=50, random_seed=0))
        with pytest.raises(NotFittedError):
            report.protected_characteristic_check(data.X, ["age_band"])

    def test_ci_ordered(self, report_fixtures):
        report, model, est, data = report_fixtures
        result = report.protected_characteristic_check(
            data.X, protected_cols=["age_band"]
        )
        assert (result["effect_lower"] <= result["effect_mean"]).all()
        assert (result["effect_mean"] <= result["effect_upper"]).all()


class TestRender:
    def test_renders_html_file(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report.html")
            report.render(output_path=path)
            assert Path(path).exists()

    def test_html_file_nonempty(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report.html")
            report.render(output_path=path)
            content = Path(path).read_text()
            assert len(content) > 100

    def test_html_has_title(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report.html")
            report.render(output_path=path)
            content = Path(path).read_text()
            assert "Bayesian Causal Forest" in content

    def test_renders_with_x(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report_x.html")
            report.render(output_path=path, X=data.X, Z=data.treatment)
            content = Path(path).read_text()
            assert "Model Configuration" in content

    def test_renders_with_protected_cols(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report_pc.html")
            report.render(
                output_path=path,
                X=data.X,
                Z=data.treatment,
                protected_cols=["age_band"],
            )
            content = Path(path).read_text()
            assert "Protected Characteristic" in content

    def test_renders_with_segment_cols(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_report_seg.html")
            report.render(
                output_path=path,
                X=data.X,
                Z=data.treatment,
                segment_cols=[["age_band"], ["channel"]],
            )
            content = Path(path).read_text()
            assert "Segment" in content

    def test_creates_parent_dir(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "nested", "report.html")
            report.render(output_path=path)
            assert Path(path).exists()

    def test_unfitted_raises(self):
        model = BayesianCausalForest()
        report = BCFAuditReport(model)
        with pytest.raises(NotFittedError):
            report.render("/tmp/should_not_exist.html")

    def test_render_without_plots(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "noplots.html")
            report.render(output_path=path, X=data.X, include_plots=False)
            content = Path(path).read_text()
            assert "Bayesian Causal Forest" in content

    def test_html_contains_methodology(self, report_fixtures):
        report, model, est, data = report_fixtures
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_method.html")
            report.render(output_path=path)
            content = Path(path).read_text()
            assert "Methodology" in content
            assert "Hahn" in content
