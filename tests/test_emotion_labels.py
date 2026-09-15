import pytest

from src.emotion_labels import EMOTION_LABELS, labels_to_target


def test_exact_ordered_six_emotion_labels():
    assert EMOTION_LABELS == ("Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust")


def test_multi_label_target_and_unknown_label_rejection():
    assert labels_to_target(["joy", "Fear"]) == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        labels_to_target(["calm"])
