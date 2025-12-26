import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "MLProj"))
from ml.data.preprocessing import (  # noqa: E402
    load_merged_data,
    show_before_preprocessing_stats,
    preprocess_text_step_by_step,
    show_after_preprocessing_stats,
    prepare_for_classical_models,
    prepare_labels,
    save_preprocessed_data,
    check_class_balance,
)


def main():
    print("Предобработка данных...")
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "processed"
    splits = ["train", "validation", "test"]
    for split in splits:
        input_file = data_dir / f"banking49_{split}.csv"
        if not input_file.exists():
            print(f"Файл {input_file} не найден, пропускаем...")
            continue
        print(f"\nОбработка {split}...")
        df = load_merged_data(input_file)
        show_before_preprocessing_stats(df)
        df_processed = preprocess_text_step_by_step(df, show_examples=False)
        show_after_preprocessing_stats(df_processed)
        df_processed, _ = prepare_for_classical_models(df_processed)
        df_processed, label_mapping = prepare_labels(df_processed)
        check_class_balance(df_processed)
        output_file = output_dir / f"preprocessed_{split}.csv"
        save_preprocessed_data(df_processed, output_file)
        if split == "train":
            import json

            mapping_file = output_dir / "preprocessed_label_mapping.json"
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print("Предобработка завершена")


if __name__ == "__main__":
    main()
