import pytest

from src.emotion import format_probabilities


def test_sigmoid_result_schema_preserves_all_six_probabilities():
    result = format_probabilities([0.7, 0.1, 0.2, 0.6, 0.1, 0.1], threshold=0.5)
    assert result["primary_emotion"] == "Joy"
    assert [item["emotion"] for item in result["predicted_emotions"]] == ["Joy", "Fear"]
    assert len(result["all_probabilities"]) == 6


def test_no_thresholded_label_is_safe_and_primary_remains_available():
    result = format_probabilities([0.2] * 6, threshold=0.8)
    assert result["predicted_emotions"] == []
    assert result["threshold_met"] is False
    assert result["primary_emotion"] == "Joy"


@pytest.mark.parametrize("text", ["", "   ", None, 42])
def test_invalid_prediction_input_is_rejected_before_model_use(text):
    from src.emotion import predict_text
    with pytest.raises(ValueError):
        predict_text(None, None, text)
