"""Thread pinning, and the cost-model fitting the calibration pilot depends on."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from budget_tune.measure.threads import (
    THREAD_VARIABLES,
    ThreadPinningError,
    declared,
    pin,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestThreadPinning:
    def test_refuses_to_pin_after_numpy_is_imported(self):
        """The failure this prevents is silent: a manifest that lies about the run.

        OpenMP and BLAS read their thread counts when the shared library loads, which
        happens on ``import numpy``. Setting the variables afterwards changes the
        environment and nothing else, so the run would record a pinned thread count it
        never had -- and CPU-seconds would carry a parallelism confound the manifest
        claims was removed.
        """
        assert "numpy" in sys.modules
        with pytest.raises(ThreadPinningError, match="already imported"):
            pin(1)

    def test_pinning_early_sets_every_backend(self):
        """Run in a fresh process, because the ordering is the thing being tested."""
        script = (
            "from budget_tune.measure.threads import pin, declared\n"
            "settings = pin(1)\n"
            "import numpy\n"
            "assert all(v == '1' for v in settings.values()), settings\n"
            "assert declared() == settings\n"
            "print('ok')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        for variable in THREAD_VARIABLES:
            env.pop(variable, None)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout

    def test_allow_late_is_available_for_the_multi_thread_comparison(self):
        before = declared()
        try:
            pin(2, allow_late=True)
            assert declared()["OMP_NUM_THREADS"] == "2"
        finally:
            for variable, value in before.items():
                if value == "<unset>":
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value

    def test_rejects_nonsense_counts(self):
        with pytest.raises(ValueError, match="at least 1"):
            pin(0, allow_late=True)

    def test_verify_reports_what_the_environment_actually_says(self):
        report = verify(1)
        assert report["requested"] == 1
        assert set(report["environment"]) == set(THREAD_VARIABLES)
        assert isinstance(report["environment_consistent"], bool)


class TestCostModel:
    """The pilot's fitting, checked against data with known exponents."""

    def _synthetic(self, exponents: dict, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        base = {"factors": 64, "epochs": 15, "n_train_interactions": 100_000}
        grid = [
            ("base", base),
            ("factors", {**base, "factors": 16}),
            ("factors", {**base, "factors": 32}),
            ("factors", {**base, "factors": 128}),
            ("epochs", {**base, "epochs": 5}),
            ("epochs", {**base, "epochs": 30}),
            ("fraction", {**base, "n_train_interactions": 25_000}),
            ("fraction", {**base, "n_train_interactions": 50_000}),
            ("corner", {"factors": 128, "epochs": 30, "n_train_interactions": 25_000}),
        ]
        for role, values in grid:
            cost = 1e-6
            for axis, exponent in exponents.items():
                cost *= values[axis] ** exponent
            cost *= float(np.exp(rng.normal(0, noise)))
            rows.append({"family": "als", "role": role, "train_cpu_seconds": cost, **values})
        return pd.DataFrame(rows)

    def test_recovers_known_exponents(self):
        from experiments.calibrate import fit_power_law

        truth = {"factors": 2.0, "epochs": 1.0, "n_train_interactions": 1.0}
        model = fit_power_law(self._synthetic(truth), "als", list(truth))

        for axis, exponent in truth.items():
            assert model["exponents"][axis] == pytest.approx(exponent, abs=1e-6)
        assert model["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_distinguishes_quadratic_from_cubic(self):
        """The one exponent the campaign estimate actually turns on."""
        from experiments.calibrate import fit_power_law

        quadratic = fit_power_law(
            self._synthetic({"factors": 2.0, "epochs": 1.0, "n_train_interactions": 1.0}),
            "als", ["factors", "epochs", "n_train_interactions"],
        )
        cubic = fit_power_law(
            self._synthetic({"factors": 3.0, "epochs": 1.0, "n_train_interactions": 1.0}),
            "als", ["factors", "epochs", "n_train_interactions"],
        )
        assert quadratic["exponents"]["factors"] == pytest.approx(2.0, abs=1e-6)
        assert cubic["exponents"]["factors"] == pytest.approx(3.0, abs=1e-6)

    def test_the_corner_is_held_out_of_the_fit(self):
        """Otherwise the corner's agreement is a residual, not a prediction.

        The corner exists to test whether the multiplicative model holds far from the base
        point. Fitting on it would guarantee it agreed and destroy the only validation the
        pilot has.
        """
        from experiments.calibrate import fit_power_law

        truth = {"factors": 2.0, "epochs": 1.0, "n_train_interactions": 1.0}
        frame = self._synthetic(truth)
        model = fit_power_law(frame, "als", list(truth))
        assert model["n_points"] == len(frame) - 1

    def test_predicts_the_held_out_corner(self):
        from experiments.calibrate import fit_power_law, predict

        truth = {"factors": 2.0, "epochs": 1.0, "n_train_interactions": 1.0}
        frame = self._synthetic(truth)
        model = fit_power_law(frame, "als", list(truth))

        corner = frame[frame.role == "corner"].iloc[0]
        assert predict(model, corner.to_dict()) == pytest.approx(
            corner["train_cpu_seconds"], rel=1e-6
        )

    def test_refuses_to_fit_an_underdetermined_model(self):
        from experiments.calibrate import fit_power_law

        frame = self._synthetic({"factors": 2.0})
        assert fit_power_law(frame.head(2), "als", ["factors", "epochs"]) is None


@pytest.mark.companion
class TestEvaluate:
    """The scoring primitive, checked against hand-computed NDCG."""

    @pytest.fixture
    def dataset(self):
        from budget_tune.companion import ensure_importable
        from budget_tune.data.catalogues import synthetic

        ensure_importable("green_rerank")
        return synthetic(fractions=(1.0,))

    def test_ndcg_matches_the_position_discount(self, dataset):
        """A hit at rank r must score 1/log2(r+1), and a miss must score zero."""
        from budget_tune.benchmark.evaluate import score

        users = dataset.eval_users()[:3]
        targets = [dataset.validation[int(u)] for u in users]

        # Place each user's target at a different rank: first, third, and nowhere.
        items = np.zeros((3, 10), dtype=np.int64)
        for position in range(3):
            filler = [i for i in range(dataset.n_items) if i != targets[position]][:10]
            items[position] = filler
        items[0][0] = targets[0]
        items[1][2] = targets[1]

        result = score(dataset, items, users, "validation", k=10)
        expected = [1.0, 1 / np.log2(4), 0.0]
        np.testing.assert_allclose(result.per_user_ndcg, expected, atol=1e-12)
        assert result.recall == pytest.approx(2 / 3)

    def test_scoring_the_wrong_split_name_is_refused(self, dataset):
        from budget_tune.benchmark.evaluate import score

        users = dataset.eval_users()[:2]
        items = np.zeros((2, 10), dtype=np.int64)
        with pytest.raises(ValueError, match="validation.*test"):
            score(dataset, items, users, "train", k=10)
