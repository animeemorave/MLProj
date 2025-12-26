import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "MLProj"))
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path
from ml.data.banking_class_mapping import create_banking77_to_banking49_mapping


def load_banking_datasets():
    print("Загрузка датасетов...")
    banking77 = load_dataset("mteb/banking77")
    banking49 = load_dataset("pilarllera/BANKING49")
    print(f"Banking77: {len(banking77['train'])} train, {len(banking77['test'])} test")
    print(f"BANKING49: {len(banking49['train'])} train, {len(banking49['test'])} test")
    return (banking77, banking49)


def create_label_mapping(banking49):
    classes49 = banking49["train"].features["labels"].names
    label_mapping = {i: name for i, name in enumerate(classes49)}
    project_root = Path(__file__).parent.parent.parent.parent
    output_path = project_root / "data" / "processed" / "banking49_label_mapping.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(label_mapping, f, indent=2)
    return label_mapping


def map_banking77_to_banking49(banking77, mapping_dict):
    print("Маппинг Banking77 -> BANKING49...")
    mapped_splits = {}
    for split in ["train", "test"]:
        df = pd.DataFrame(banking77[split])
        initial_count = len(df)
        df["mapped_label"] = df["label_text"].map(mapping_dict)
        df_mapped = df[df["mapped_label"].notna()].copy()
        dropped_count = initial_count - len(df_mapped)
        print(f"  {split}: {len(df_mapped)}/{initial_count} ({dropped_count / initial_count * 100:.1f}% отброшено)")
        mapped_splits[split] = df_mapped
    return mapped_splits


def merge_with_banking49(banking77_mapped, banking49):
    print("Объединение датасетов...")
    classes49 = banking49["train"].features["labels"].names
    class_to_id = {name: idx for idx, name in enumerate(classes49)}
    merged_splits = {}
    for split in ["train", "test"]:
        df77 = banking77_mapped[split].copy()
        df49 = pd.DataFrame(banking49[split])
        df77["label_id"] = df77["mapped_label"].map(class_to_id)
        df49["label_id"] = df49["labels"]
        df77_final = df77[["text", "label_id"]].rename(columns={"label_id": "label"})
        df49_final = df49[["text", "label_id"]].rename(columns={"label_id": "label"})
        merged_df = pd.concat([df77_final, df49_final], ignore_index=True)
        initial_size = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=["text"], keep="first")
        removed = initial_size - len(merged_df)
        print(f"  {split}: {len(merged_df)} примеров ({removed} дубликатов удалено)")
        merged_splits[split] = merged_df
    return merged_splits


def save_merged_data(merged_splits, label_mapping):
    project_root = Path(__file__).parent.parent.parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Сохранение данных...")
    for split, df in merged_splits.items():
        output_path = output_dir / f"banking49_{split}.csv"
        df.to_csv(output_path, index=False)
        print(f"  {split}: {len(df)} примеров, {df['label'].nunique()} классов -> {output_path}")


def main():
    banking77, banking49 = load_banking_datasets()
    label_mapping = create_label_mapping(banking49)
    mapping_dict = create_banking77_to_banking49_mapping()
    banking77_mapped = map_banking77_to_banking49(banking77, mapping_dict)
    merged_splits = merge_with_banking49(banking77_mapped, banking49)
    save_merged_data(merged_splits, label_mapping)
    print("Готово!")


if __name__ == "__main__":
    main()
