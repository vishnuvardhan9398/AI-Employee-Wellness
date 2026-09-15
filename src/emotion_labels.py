"""The single, ordered label definition used by all Milestone 2 modules."""

EMOTION_LABELS = ("Joy", "Sadness", "Anger", "Fear", "Surprise", "Disgust")
LABEL_TO_ID = {label: index for index, label in enumerate(EMOTION_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def empty_target() -> list[float]:
    """Return an all-negative six-label target vector."""
    return [0.0] * len(EMOTION_LABELS)


def labels_to_target(labels) -> list[float]:
    """Convert a case-insensitive iterable of project labels into a multi-hot vector."""
    target = empty_target()
    normalized = {str(label).strip().casefold() for label in labels}
    unknown = normalized - {label.casefold() for label in EMOTION_LABELS}
    if unknown:
        raise ValueError(f"Unsupported emotion label(s): {sorted(unknown)}")
    for label, index in LABEL_TO_ID.items():
        if label.casefold() in normalized:
            target[index] = 1.0
    return target
