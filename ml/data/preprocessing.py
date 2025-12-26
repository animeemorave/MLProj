import pandas as pd
import re
from typing import Tuple, Dict
from pathlib import Path


def load_merged_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df


def show_before_preprocessing_stats(df: pd.DataFrame):
    text_lengths = df["text"].str.len()
    print(f"До обработки: {len(df)} примеров, {df['label'].nunique()} классов, средняя длина текста: {text_lengths.mean():.0f} символов")


def lowercase_text(text: str) -> str:
    return text.lower()


def remove_extra_spaces(text: str) -> str:
    return re.sub("\\s+", " ", text).strip()


def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
    if keep_punctuation:
        text = re.sub("[^a-zA-Z0-9\\s\\.,!?;:\\-\\'\"]", "", text)
    else:
        text = re.sub("[^a-zA-Z0-9\\s]", "", text)
    return text


def preprocess_text_step_by_step(
    df: pd.DataFrame, show_examples: bool = True
) -> pd.DataFrame:
    df_processed = df.copy()
    df_processed["text"] = df_processed["text"].apply(lowercase_text)
    df_processed["text"] = df_processed["text"].apply(remove_extra_spaces)
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed["text"].str.strip() != ""]
    removed_count = initial_count - len(df_processed)
    if removed_count > 0:
        print(f"Удалено {removed_count} пустых строк")
    return df_processed


def show_after_preprocessing_stats(df: pd.DataFrame):
    text_lengths = df["text"].str.len()
    print(f"После обработки: {len(df)} примеров, средняя длина текста: {text_lengths.mean():.0f} символов")


def prepare_for_classical_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    return (df, {})


def prepare_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if df["label"].dtype == "object":
        unique_labels = sorted(df["label"].unique())
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        df["label"] = df["label"].map(label_to_id)
    else:
        unique_labels = sorted(df["label"].unique())
        label_to_id = {str(label): int(label) for label in unique_labels}
        id_to_label = {int(label): str(label) for label in unique_labels}
    label_mapping = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_classes": len(label_to_id),
    }
    return (df, label_mapping)


def save_preprocessed_data(df: pd.DataFrame, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Сохранено: {output_path} ({len(df)} строк)")


def check_class_balance(df: pd.DataFrame):
    label_counts = df["label"].value_counts().sort_index()
    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"Классов: {len(label_counts)}, дисбаланс: {imbalance_ratio:.1f}x")
