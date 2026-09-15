import pytest


def test_dynamic_multilabel_metrics_and_selection_rule():
    pytest.importorskip("sklearn")
    from src.evaluation import calculate_metrics, compare_models, comparison_rows
    truth = [[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]]
    scores = [[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.8, 0.7, 0.1, 0.1]]
    metrics = calculate_metrics(truth, scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] >= 0.0
    assert compare_models({"bert": metrics, "distilbert": {**metrics, "macro_f1": 0.0}})["best_model"] == "bert"
    assert comparison_rows({"bert": metrics})[0]["model"] == "bert"
