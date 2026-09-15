"""Optional real-model smoke test: never runs in the normal fast test suite."""

import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_TRANSFORMER_INTEGRATION") != "1", reason="set RUN_TRANSFORMER_INTEGRATION=1 to download/load real models")
@pytest.mark.parametrize("model_kind", ["bert", "distilbert"])
def test_real_model_can_be_saved_loaded_and_predicts_six_probabilities(tmp_path, model_kind):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.emotion import create_model, load_local_model, predict_text, save_model

    model, tokenizer = create_model(model_kind)
    saved = save_model(model, tokenizer, tmp_path / model_kind)
    loaded_model, loaded_tokenizer = load_local_model(saved)
    result = predict_text(loaded_model, loaded_tokenizer, "I feel excited but nervous about tomorrow.")
    assert len(result["all_probabilities"]) == 6
    assert all(0.0 <= value <= 1.0 for value in result["all_probabilities"].values())
