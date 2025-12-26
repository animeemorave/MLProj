import sys
from pathlib import Path
import joblib
import json

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "project"))
from ml.data.preprocessing import lowercase_text, remove_extra_spaces  # noqa: E402


def load_model_and_vectorizer(
    model_name: str = "logistic_regression", verbose: bool = False
):
    models_dir = project_root / "models"
    model_path = models_dir / f"{model_name}_model.pkl"
    vectorizer_path = models_dir / f"{model_name}_vectorizer.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Векторизатор не найден: {vectorizer_path}")
    if verbose:
        print(f"Загрузка модели: {model_path}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    if verbose:
        print("Модель и векторизатор загружены")
    return (model, vectorizer)


def load_label_mapping(verbose: bool = False):
    banking49_mapping_file = (
        project_root / "data" / "processed" / "banking49_label_mapping.json"
    )
    if banking49_mapping_file.exists():
        with open(banking49_mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        id_to_label = {str(k): str(v) for k, v in mapping_data.items()}
        return {"id_to_label": id_to_label}
    if verbose:
        print("Маппинг меток не найден, используем числовые метки")
    return None


def preprocess_text(text: str) -> str:
    text = lowercase_text(text)
    text = remove_extra_spaces(text)
    return text


def predict_text(
    model,
    vectorizer,
    text: str,
    label_mapping=None,
    threshold: float = 0.2,
    top_n: int = 10,
    verbose: bool = False,
):
    text_processed = preprocess_text(text)
    text_vec = vectorizer.transform([text_processed])
    probabilities = model.predict_proba(text_vec)[0]
    predicted_class = model.predict(text_vec)[0]
    max_probability = probabilities.max()

    top_n_indices = probabilities.argsort()[-top_n:][::-1]
    top_n_classes = [
        {
            "class": int(idx),
            "probability": float(probabilities[idx]),
            "name": label_mapping["id_to_label"].get(str(idx), f"Класс {idx}")
            if label_mapping and "id_to_label" in label_mapping
            else f"Класс {idx}",
        }
        for idx in top_n_indices
    ]

    is_out_of_scope = max_probability < threshold

    if verbose:
        if label_mapping and "id_to_label" in label_mapping:
            class_name = label_mapping["id_to_label"].get(
                str(predicted_class), f"Класс {predicted_class}"
            )
            print(f"Предсказанный класс: {predicted_class} - {class_name}")
        else:
            print(f"Предсказанный класс: {predicted_class}")
        print(f"Вероятность: {max_probability:.2%}")
        print(f"\nТоп-{top_n} наиболее вероятных классов:")
        for i, cls_info in enumerate(top_n_classes, 1):
            print(f"  {i}. {cls_info['name']}: {cls_info['probability']:.2%}")
        if is_out_of_scope:
            print(
                f"\n⚠️  Низкая уверенность ({max_probability:.2%} < {threshold:.2%}) - возможно out-of-scope"
            )

    return {
        "predicted_class": int(predicted_class),
        "probability": float(max_probability),
        "is_out_of_scope": is_out_of_scope,
        "top_n_classes": top_n_classes,
    }


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if verbose:
        sys.argv = [arg for arg in sys.argv if arg not in ["--verbose", "-v"]]

    model_name = "logistic_regression"
    if len(sys.argv) > 1 and sys.argv[1] in ["--model", "-m"] and (len(sys.argv) > 2):
        model_name = sys.argv[2]
        text = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        if verbose:
            print("=" * 60)
            print("Предсказание намерения из текста")
            print("=" * 60)
        print("\nВведите текст для классификации:")
        print("(или запустите скрипт с текстом: python predict.py 'ваш текст')")
        print(
            "(или выберите модель: python predict.py --model random_forest 'ваш текст')"
        )
        text = input("> ")

    if not text.strip():
        print("Ошибка: текст не может быть пустым")
        return

    try:
        model, vectorizer = load_model_and_vectorizer(model_name, verbose=verbose)
        label_mapping = load_label_mapping(verbose=verbose)
        result = predict_text(
            model,
            vectorizer,
            text,
            label_mapping,
            threshold=0.2,
            top_n=10,
            verbose=verbose,
        )

        if verbose:
            print("\n" + "=" * 60)
            print("Итоговый результат")
            print("=" * 60)
            print(f"Текст: {text}")
            if label_mapping and "id_to_label" in label_mapping:
                class_name = label_mapping["id_to_label"].get(
                    str(result["predicted_class"]), "Неизвестно"
                )
                print(f"Намерение: {class_name}")
            print(f"Уверенность: {result['probability']:.2%}")
            if result["is_out_of_scope"]:
                print("Статус: OUT-OF-SCOPE (низкая уверенность)")
            else:
                print("Статус: Распознано")
        else:
            if label_mapping and "id_to_label" in label_mapping:
                class_name = label_mapping["id_to_label"].get(
                    str(result["predicted_class"]), "Неизвестно"
                )
                print(f"{class_name} ({result['probability']:.1%})")
            else:
                print(
                    f"Класс {result['predicted_class']} ({result['probability']:.1%})"
                )

            if result["is_out_of_scope"]:
                print("⚠️  Низкая уверенность")

            print("\nТоп-10 наиболее вероятных классов:")
            for i, cls_info in enumerate(result["top_n_classes"], 1):
                print(f"  {i}. {cls_info['name']}: {cls_info['probability']:.2%}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nСначала нужно обучить модель:")
        print("  python ml/scripts/train_with_cv.py")
    except Exception as e:
        print(f"Ошибка: {e}")
        if verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
