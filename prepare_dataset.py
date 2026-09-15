from pathlib import Path
import pandas as pd
from datasets import load_dataset


# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Your project's exact required column order
OUTPUT_COLUMNS = [
    "text",
    "Joy",
    "Sadness",
    "Anger",
    "Fear",
    "Surprise",
    "Disgust",
]


def convert_split(split):
    """Convert GoEmotions Ekman labels to this project's 6-column format."""

    rows = []

    for item in split:
        text = str(item["text"]).strip()
        labels = item["labels"]

        # Skip blank text
        if not text:
            continue

        # GoEmotions simplified Ekman label mapping
        # 0 = anger
        # 1 = disgust
        # 2 = fear
        # 3 = joy
        # 4 = sadness
        # 5 = surprise
        # 6 = neutral

        row = {
            "text": text,
            "Joy": 0,
            "Sadness": 0,
            "Anger": 0,
            "Fear": 0,
            "Surprise": 0,
            "Disgust": 0,
        }

        for label in labels:
            if label == 0:
                row["Anger"] = 1
            elif label == 1:
                row["Disgust"] = 1
            elif label == 2:
                row["Fear"] = 1
            elif label == 3:
                row["Joy"] = 1
            elif label == 4:
                row["Sadness"] = 1
            elif label == 5:
                row["Surprise"] = 1

        # Keep only rows containing at least one
        # of the six required emotions.
        emotion_values = [
            row["Joy"],
            row["Sadness"],
            row["Anger"],
            row["Fear"],
            row["Surprise"],
            row["Disgust"],
        ]

        if sum(emotion_values) > 0:
            rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    print("Downloading GoEmotions dataset...")
    
    dataset = load_dataset(
        "AiLab-IMCS-UL/go_emotions-en",
        "simplified_ekman"
    )

    split_mapping = {
        "train": "emotion_train.csv",
        "validation": "emotion_validation.csv",
        "test": "emotion_test.csv",
    }

    for split_name, file_name in split_mapping.items():
        print(f"\nProcessing {split_name} split...")

        df = convert_split(dataset[split_name])

        output_path = OUTPUT_DIR / file_name
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Saved: {output_path}")
        print(f"Samples: {len(df)}")
        print(df.head(3).to_string(index=False))

    print("\nDataset preparation completed successfully!")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()