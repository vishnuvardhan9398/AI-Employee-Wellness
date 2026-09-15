from datasets import load_dataset
import csv
from pathlib import Path

# Our six target emotions
TARGET_LABELS = [
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust"
]

# GoEmotions label mapping
# We keep the mapping explicit and compatible with our 6-label project.
EMOTION_MAP = {
    "admiration": "Joy",
    "amusement": "Joy",
    "approval": "Joy",
    "caring": "Joy",
    "desire": "Joy",
    "excitement": "Joy",
    "gratitude": "Joy",
    "joy": "Joy",
    "love": "Joy",
    "optimism": "Joy",
    "pride": "Joy",
    "relief": "Joy",

    "sadness": "Sadness",
    "disappointment": "Sadness",
    "embarrassment": "Sadness",
    "grief": "Sadness",
    "remorse": "Sadness",

    "anger": "Anger",
    "annoyance": "Anger",
    "disapproval": "Anger",

    "fear": "Fear",
    "nervousness": "Fear",

    "surprise": "Surprise",
    "realization": "Surprise",
    "confusion": "Surprise",
    "curiosity": "Surprise",

    "disgust": "Disgust",
}

OUTPUT_DIR = Path("data/training_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_split(dataset_split, output_file, label_names):
    rows = []
    counts = {label: 0 for label in TARGET_LABELS}

    for item in dataset_split:
        text = (item["text"] or "").strip()

        if not text:
            continue

        # Convert numeric GoEmotions labels to names
        emotion_names = [
            label_names[label_id]
            for label_id in item["labels"]
        ]

        # Map original labels to our 6 labels
        mapped = []

        for emotion in emotion_names:
            if emotion in EMOTION_MAP:
                mapped.append(EMOTION_MAP[emotion])

        mapped = list(set(mapped))

        # Keep only samples that map to exactly ONE of our six emotions.
        # This creates a cleaner initial training dataset.
        if len(mapped) != 1:
            continue

        target_emotion = mapped[0]

        row = {"text": text}

        for label in TARGET_LABELS:
            row[label] = 1 if label == target_emotion else 0

        rows.append(row)
        counts[target_emotion] += 1

    output_path = OUTPUT_DIR / output_file

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text"] + TARGET_LABELS
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_path}")
    print(f"Total samples: {len(rows)}")

    print("Emotion distribution:")
    for label, count in counts.items():
        print(f"  {label:<10}: {count}")

    return rows


print("Downloading GoEmotions dataset...")
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

# Get original GoEmotions label names
label_names = dataset["train"].features["labels"].feature.names

print("\nPreparing TRAIN split...")
prepare_split(
    dataset["train"],
    "emotion_train.csv",
    label_names
)

print("\nPreparing VALIDATION split...")
prepare_split(
    dataset["validation"],
    "emotion_validation.csv",
    label_names
)

print("\nPreparing TEST split...")
prepare_split(
    dataset["test"],
    "emotion_test.csv",
    label_names
)

print("\nSUCCESS! New V2 dataset is ready.")