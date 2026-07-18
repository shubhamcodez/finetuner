from __future__ import annotations

import json

import numpy as np
import pytest

from finetuner.analysis.artifacts import LayerAnalysis, RepresentationPoint, write_analysis_artifact
from finetuner.analysis.reducers import linear_cka, pca_2d, reduce_2d


def test_pca_is_deterministic_and_centered():
    values = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]], dtype=float)
    first, metadata = pca_2d(values)
    second, _ = reduce_2d(values, "pca", seed=999)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.mean(axis=0), [0, 0], atol=1e-10)
    assert metadata["explained_variance_ratio"][0] > 0.99


def test_linear_cka_detects_identical_representation_geometry():
    values = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    assert linear_cka(values, values) == pytest.approx(1.0)


def test_analysis_artifact_schema(tmp_path):
    path = write_analysis_artifact(
        tmp_path / "representations.json",
        model="model",
        reducer="pca",
        pooling="mean",
        layers=[
            LayerAnalysis(
                0,
                [RepresentationPoint("0", "math", "2+2", 0.0, 1.0)],
                1.0,
                0.1,
            )
        ],
        cka=[[1.0]],
    )
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["schema_version"] == 1
    assert payload["layers"][0]["points"][0]["label"] == "math"
