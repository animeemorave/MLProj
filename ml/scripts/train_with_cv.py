import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "project"))
from ml.models.classical_models import (  # noqa: E402
    load_preprocessed_data,
    create_tfidf_vectorizer,
    train_final_model_with_cv,
    save_model,
    save_oof_predictions,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


def train_logistic_regression_cv():
    print("Обучение Logistic Regression...")
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_preprocessed_data(data_dir / "banking49_train.csv")
    X_val, y_val = load_preprocessed_data(data_dir / "banking49_validation.csv")
    X_test, y_test = load_preprocessed_data(data_dir / "banking49_test.csv")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    vectorizer = create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_val_vec = vectorizer.fit_transform(X_train_val)
    X_test_vec = vectorizer.transform(X_test)
    model_params = {"max_iter": 1000, "C": 1.0, "random_state": 42, "solver": "lbfgs"}
    final_model, oof_pred, oof_proba, fold_accs, cv_mean, cv_std, test_acc = (
        train_final_model_with_cv(
            LogisticRegression,
            X_train_val_vec,
            y_train_val,
            X_test_vec,
            y_test,
            model_params=model_params,
            model_name="Logistic Regression",
            n_splits=5,
            calibration_method="isotonic",
        )
    )
    oof_path = results_dir / "logistic_regression_oof_predictions.csv"
    save_oof_predictions(
        oof_pred,
        oof_proba,
        X_train_val.tolist(),
        y_train_val.tolist(),
        oof_path,
        "logistic_regression",
    )
    model_path, vectorizer_path = save_model(
        final_model, vectorizer, "logistic_regression", models_dir
    )
    print(f"Модель сохранена: {model_path}")


def train_random_forest_cv():
    print("Обучение Random Forest...")
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_preprocessed_data(data_dir / "banking49_train.csv")
    X_val, y_val = load_preprocessed_data(data_dir / "banking49_validation.csv")
    X_test, y_test = load_preprocessed_data(data_dir / "banking49_test.csv")
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    vectorizer = create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_val_vec = vectorizer.fit_transform(X_train_val)
    X_test_vec = vectorizer.transform(X_test)
    model_params = {
        "n_estimators": 150,
        "max_depth": 18,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    final_model, oof_pred, oof_proba, fold_accs, cv_mean, cv_std, test_acc = (
        train_final_model_with_cv(
            RandomForestClassifier,
            X_train_val_vec,
            y_train_val,
            X_test_vec,
            y_test,
            model_params=model_params,
            model_name="Random Forest",
            n_splits=5,
            calibration_method="isotonic",
        )
    )
    oof_path = results_dir / "random_forest_oof_predictions.csv"
    save_oof_predictions(
        oof_pred,
        oof_proba,
        X_train_val.tolist(),
        y_train_val.tolist(),
        oof_path,
        "random_forest",
    )
    model_path, vectorizer_path = save_model(
        final_model, vectorizer, "random_forest", models_dir
    )
    print(f"Модель сохранена: {model_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Обучение модели с 5-fold cross-validation и калибровкой"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lr", "rf", "both"],
        default="both",
        help="Модель для обучения: lr (Logistic Regression), rf (Random Forest), both (обе)",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["isotonic", "sigmoid"],
        default="isotonic",
        help="Метод калибровки: isotonic (лучше для сложных распределений) или sigmoid (быстрее)",
    )
    args = parser.parse_args()
    if args.model in ["lr", "both"]:
        train_logistic_regression_cv()
    if args.model in ["rf", "both"]:
        train_random_forest_cv()


if __name__ == "__main__":
    main()
