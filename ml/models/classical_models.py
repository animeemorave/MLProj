import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path
from typing import Tuple, Dict


def load_preprocessed_data(data_path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["label"]
    return (X, y)


def create_tfidf_vectorizer(
    max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)
):
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range, stop_words="english"
    )
    return vectorizer


def vectorize_texts(
    vectorizer: TfidfVectorizer, X_train: pd.Series, X_val: pd.Series, X_test: pd.Series
):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    return (X_train_vec, X_val_vec, X_test_vec)


def train_logistic_regression(
    X_train_vec, y_train, X_val_vec, y_val, max_iter: int = 1000, C: float = 1.0
):
    model = LogisticRegression(max_iter=max_iter, C=C, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)
    y_val_pred = model.predict(X_val_vec)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return (model, val_accuracy)


def train_naive_bayes(X_train_vec, y_train, X_val_vec, y_val):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_val_pred = model.predict(X_val_vec)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return (model, val_accuracy)


def train_svm(
    X_train_vec, y_train, X_val_vec, y_val, kernel: str = "linear", C: float = 1.0
):
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    model.fit(X_train_vec, y_train)
    y_val_pred = model.predict(X_val_vec)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return (model, val_accuracy)


def train_random_forest(
    X_train_vec,
    y_train,
    X_val_vec,
    y_val,
    n_estimators: int = 100,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: str = "balanced",
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_vec, y_train)
    y_val_pred = model.predict(X_val_vec)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    return (model, val_accuracy)


def evaluate_model(model, X_test_vec, y_test, model_name: str):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    return (accuracy, y_pred)


def cross_validate_model(
    model_class,
    X_train_val_vec,
    y_train_val,
    n_splits: int = 5,
    model_params: Dict = None,
    model_name: str = "Model",
):
    if model_params is None:
        model_params = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y_train_val), dtype=int)
    oof_probabilities = None
    fold_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train_val_vec, y_train_val), 1
    ):
        X_train_fold = X_train_val_vec[train_idx]
        y_train_fold = (
            y_train_val.iloc[train_idx]
            if isinstance(y_train_val, pd.Series)
            else y_train_val[train_idx]
        )
        X_val_fold = X_train_val_vec[val_idx]
        y_val_fold = (
            y_train_val.iloc[val_idx]
            if isinstance(y_train_val, pd.Series)
            else y_train_val[val_idx]
        )
        model = model_class(**model_params)
        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)
        fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
        fold_accuracies.append(fold_accuracy)
        oof_predictions[val_idx] = y_val_pred
        if hasattr(model, "predict_proba"):
            val_proba = model.predict_proba(X_val_fold)
            if oof_probabilities is None:
                n_classes = val_proba.shape[1]
                oof_probabilities = np.zeros((len(y_train_val), n_classes))
            oof_probabilities[val_idx] = val_proba
        print(f"Fold {fold}/{n_splits}: {fold_accuracy:.4f} ({fold_accuracy * 100:.2f}%)")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"CV {model_name}: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    return (
        oof_predictions,
        oof_probabilities,
        fold_accuracies,
        mean_accuracy,
        std_accuracy,
    )


def train_final_model_with_cv(
    model_class,
    X_train_val_vec,
    y_train_val,
    X_test_vec,
    y_test,
    model_params: Dict = None,
    model_name: str = "Model",
    n_splits: int = 5,
    calibration_method: str = "isotonic",
):
    print(f"\nОбучение {model_name}...")
    if model_params is None:
        model_params = {}
    oof_predictions, oof_probabilities, fold_accuracies, cv_mean, cv_std = (
        cross_validate_model(
            model_class,
            X_train_val_vec,
            y_train_val,
            n_splits,
            model_params,
            model_name,
        )
    )
    from sklearn.model_selection import train_test_split

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_val_vec,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val,
    )
    base_model = model_class(**model_params)
    base_model.fit(X_train, y_train)
    calibrated_model = calibrate_model(
        base_model, X_train, y_train, X_cal, y_cal, method=calibration_method
    )
    test_accuracy, y_test_pred = evaluate_model(
        calibrated_model, X_test_vec, y_test, model_name
    )
    base_proba = base_model.predict_proba(X_test_vec)
    cal_proba = calibrated_model.predict_proba(X_test_vec)
    base_max_proba = np.max(base_proba, axis=1)
    cal_max_proba = np.max(cal_proba, axis=1)
    print(f"Test {model_name}: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Уверенность: {cal_max_proba.mean():.4f} (было {base_max_proba.mean():.4f})")
    return (
        calibrated_model,
        oof_predictions,
        oof_probabilities,
        fold_accuracies,
        cv_mean,
        cv_std,
        test_accuracy,
    )


def save_oof_predictions(
    oof_predictions,
    oof_probabilities,
    texts,
    labels,
    output_path: Path,
    model_name: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame(
        {"text": texts, "true_label": labels, "oof_prediction": oof_predictions}
    )
    if oof_probabilities is not None:
        for i in range(oof_probabilities.shape[1]):
            df_oof[f"oof_probability_class_{i}"] = oof_probabilities[:, i]
    df_oof.to_csv(output_path, index=False)
    return output_path


def calibrate_model(
    base_model,
    X_train_vec,
    y_train,
    X_val_vec,
    y_val,
    method: str = "isotonic",
    cv: int = 5,
):
    from sklearn.calibration import FrozenEstimator

    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model), method=method
    )
    calibrated_model.fit(X_val_vec, y_val)
    return calibrated_model


def train_final_model_with_calibration(
    model_class,
    X_train_val_vec,
    y_train_val,
    X_test_vec,
    y_test,
    model_params: Dict = None,
    model_name: str = "Model",
    n_splits: int = 5,
    calibration_method: str = "isotonic",
):
    print("\n" + "=" * 60)
    print(f"Обучение с калибровкой: {model_name}")
    print("=" * 60)
    if model_params is None:
        model_params = {}
    from sklearn.model_selection import train_test_split

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_val_vec,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val,
    )
    print("\nРазделение данных:")
    print(f"  Обучение базовой модели: {X_train.shape[0]} примеров")
    print(f"  Калибровка: {X_cal.shape[0]} примеров")
    print("\nОбучение базовой модели...")
    base_model = model_class(**model_params)
    base_model.fit(X_train, y_train)
    print("Базовая модель обучена")
    calibrated_model = calibrate_model(
        base_model, X_train, y_train, X_cal, y_cal, method=calibration_method
    )
    print("\n" + "=" * 60)
    print("Оценка на Test данных")
    print("=" * 60)
    test_accuracy, y_test_pred = evaluate_model(
        calibrated_model, X_test_vec, y_test, model_name
    )
    print("\n" + "=" * 60)
    print("Сравнение вероятностей до и после калибровки")
    print("=" * 60)
    base_proba = base_model.predict_proba(X_test_vec)
    cal_proba = calibrated_model.predict_proba(X_test_vec)
    base_max_proba = np.max(base_proba, axis=1)
    cal_max_proba = np.max(cal_proba, axis=1)
    print("\nМаксимальная вероятность (до калибровки):")
    print(f"  Среднее: {base_max_proba.mean():.4f}")
    print(f"  Медиана: {np.median(base_max_proba):.4f}")
    print(f"  Мин: {base_max_proba.min():.4f}, Макс: {base_max_proba.max():.4f}")
    print("\nМаксимальная вероятность (после калибровки):")
    print(f"  Среднее: {cal_max_proba.mean():.4f}")
    print(f"  Медиана: {np.median(cal_max_proba):.4f}")
    print(f"  Мин: {cal_max_proba.min():.4f}, Макс: {cal_max_proba.max():.4f}")
    print(
        f"\nИзменение уверенности: {(cal_max_proba.mean() - base_max_proba.mean()) * 100:+.2f} процентных пунктов"
    )
    return (calibrated_model, test_accuracy, cal_proba)


def save_model(model, vectorizer, model_name: str, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}_model.pkl"
    vectorizer_path = output_dir / f"{model_name}_vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    return (model_path, vectorizer_path)
