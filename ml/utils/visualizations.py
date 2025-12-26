import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List
import json

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_label_mapping(mapping_path: Path) -> Dict[int, str]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def plot_metrics_comparison(
    metrics_lr: Dict[str, float], metrics_rf: Dict[str, float], save_path: Path
):
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    lr_values = [
        metrics_lr["accuracy"],
        metrics_lr["precision"],
        metrics_lr["recall"],
        metrics_lr["f1"],
    ]
    rf_values = [
        metrics_rf["accuracy"],
        metrics_rf["precision"],
        metrics_rf["recall"],
        metrics_rf["f1"],
    ]
    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        lr_values,
        width,
        label="Logistic Regression",
        color="#2ecc71",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        rf_values,
        width,
        label="Random Forest",
        color="#e74c3c",
        alpha=0.8,
    )
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Сравнение метрик моделей", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_folds_comparison(
    fold_accs_lr: List[float], fold_accs_rf: List[float], save_path: Path
):
    folds = [f"Fold {i + 1}" for i in range(len(fold_accs_lr))]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(folds))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        [acc * 100 for acc in fold_accs_lr],
        width,
        label="Logistic Regression",
        color="#2ecc71",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        [acc * 100 for acc in fold_accs_rf],
        width,
        label="Random Forest",
        color="#e74c3c",
        alpha=0.8,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Точность моделей по фолдам (5-Fold CV)", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_top_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: Dict[int, str],
    top_n: int = 10,
    model_name: str = "Model",
    save_path: Path = None,
):
    cm = confusion_matrix(y_true, y_pred)
    class_counts = pd.Series(y_true).value_counts().sort_values(ascending=False)
    top_classes = class_counts.head(top_n).index.tolist()
    cm_top = cm[np.ix_(top_classes, top_classes)]
    class_names = [label_mapping.get(cls, f"Class {cls}") for cls in top_classes]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_top,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Количество примеров"},
    )
    ax.set_title(
        f"Confusion Matrix (Top {top_n} классов) - {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Предсказанный класс", fontsize=12, fontweight="bold")
    ax.set_ylabel("Истинный класс", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_class_distribution(
    y: np.ndarray,
    label_mapping: Dict[int, str],
    top_n: int = 15,
    save_path: Path = None,
):
    class_counts = pd.Series(y).value_counts().sort_values(ascending=False)
    top_classes = class_counts.head(top_n)
    class_names = [label_mapping.get(cls, f"Class {cls}") for cls in top_classes.index]
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        range(len(top_classes)), top_classes.values, color="#3498db", alpha=0.8
    )
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Количество примеров", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Распределение классов (Top {top_n})", fontsize=14, fontweight="bold", pad=20
    )
    ax.grid(axis="x", alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, top_classes.values)):
        ax.text(
            count + max(top_classes.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count}",
            ha="left",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"График сохранен: {save_path}")
    else:
        plt.show()


def plot_roc_curve_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: Path = None,
    n_classes: int = None,
):
    if n_classes is None:
        n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["macro"] = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    tpr["macro"] = np.zeros_like(fpr["macro"])
    for i in range(n_classes):
        tpr["macro"] += np.interp(fpr["macro"], fpr[i], tpr[i])
    tpr["macro"] /= n_classes
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average ROC (AUC = {roc_auc['macro']:.3f})",
        color="navy",
        linestyle="--",
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"ROC Curve (Macro-average) - {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return roc_auc["macro"]


def plot_precision_recall_curve_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: Path = None,
    n_classes: int = None,
):
    if n_classes is None:
        n_classes = y_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_proba[:, i]
        )
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_proba[:, i])
    precision["macro"] = np.unique(
        np.concatenate([precision[i] for i in range(n_classes)])
    )
    recall["macro"] = np.zeros_like(precision["macro"])
    for i in range(n_classes):
        recall["macro"] += np.interp(
            precision["macro"], precision[i][::-1], recall[i][::-1]
        )
    recall["macro"] /= n_classes
    average_precision["macro"] = np.mean(list(average_precision.values()))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        recall["macro"],
        precision["macro"],
        label=f"Macro-average PR (AP = {average_precision['macro']:.3f})",
        color="navy",
        linestyle="--",
        linewidth=2,
    )
    baseline = np.sum(y_true_bin, axis=0) / len(y_true)
    baseline = np.mean(baseline)
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=1,
        label=f"Baseline (AP = {baseline:.3f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Precision-Recall Curve (Macro-average) - {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return average_precision["macro"]


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None,
    n_bins: int = 10,
    y_proba_before: np.ndarray = None,
):
    from sklearn.calibration import calibration_curve

    max_proba = np.max(y_proba, axis=1)
    y_pred = np.argmax(y_proba, axis=1)
    y_binary = (y_pred == y_true).astype(int)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary, max_proba, n_bins=n_bins, strategy="quantile"
    )
    if y_proba_before is not None:
        max_proba_before = np.max(y_proba_before, axis=1)
        y_pred_before = np.argmax(y_proba_before, axis=1)
        y_binary_before = (y_pred_before == y_true).astype(int)
        fraction_of_positives_before, mean_predicted_value_before = calibration_curve(
            y_binary_before, max_proba_before, n_bins=n_bins, strategy="quantile"
        )
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax1, ax2, ax3 = axes
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    if y_proba_before is not None:
        ax1.plot(
            mean_predicted_value_before,
            fraction_of_positives_before,
            "o-",
            label="До калибровки",
            color="red",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="После калибровки" if y_proba_before is not None else f"{model_name}",
        color="blue",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        [0, 1], [0, 1], "k--", label="Идеально откалиброванная модель", linewidth=2
    )
    ax1.set_xlabel("Средняя предсказанная вероятность", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Доля правильных предсказаний", fontsize=12, fontweight="bold")
    ax1.set_title("Диаграмма надёжности", fontsize=14, fontweight="bold", pad=20)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    if y_proba_before is not None:
        ax2.hist(
            max_proba_before,
            bins=n_bins,
            edgecolor="black",
            alpha=0.5,
            color="red",
            label="До калибровки",
            density=True,
        )
        ax2.hist(
            max_proba,
            bins=n_bins,
            edgecolor="black",
            alpha=0.5,
            color="blue",
            label="После калибровки",
            density=True,
        )
        ax2.legend(fontsize=11)
    else:
        ax2.hist(
            max_proba, bins=n_bins, edgecolor="black", alpha=0.7, color="steelblue"
        )
    ax2.set_xlabel("Предсказанная вероятность", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Плотность", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Распределение максимальных вероятностей",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax2.grid(alpha=0.3, axis="y")
    if y_proba_before is not None:
        ax3.axis("off")
        stats_text = f"\nСтатистика уверенности модели:\n\nДО КАЛИБРОВКИ:\n  Среднее: {max_proba_before.mean():.4f}\n  Медиана: {np.median(max_proba_before):.4f}\n  Мин: {max_proba_before.min():.4f}\n  Макс: {max_proba_before.max():.4f}\n\nПОСЛЕ КАЛИБРОВКИ:\n  Среднее: {max_proba.mean():.4f}\n  Медиана: {np.median(max_proba):.4f}\n  Мин: {max_proba.min():.4f}\n  Макс: {max_proba.max():.4f}\n\nИЗМЕНЕНИЕ:\n  Среднее: {(max_proba.mean() - max_proba_before.mean()) * 100:+.2f} п.п.\n  Медиана: {(np.median(max_proba) - np.median(max_proba_before)) * 100:+.2f} п.п.\n        "
        ax3.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
            transform=ax3.transAxes,
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_reliability_diagram_comparison(
    y_true: np.ndarray,
    y_proba_1: np.ndarray,
    model_name_1: str,
    y_proba_2: np.ndarray,
    model_name_2: str,
    save_path: Path = None,
    n_bins: int = 10,
):
    from sklearn.calibration import calibration_curve

    max_proba_1 = np.max(y_proba_1, axis=1)
    y_pred_1 = np.argmax(y_proba_1, axis=1)
    y_binary_1 = (y_pred_1 == y_true).astype(int)
    fraction_of_positives_1, mean_predicted_value_1 = calibration_curve(
        y_binary_1, max_proba_1, n_bins=n_bins, strategy="quantile"
    )
    max_proba_2 = np.max(y_proba_2, axis=1)
    y_pred_2 = np.argmax(y_proba_2, axis=1)
    y_binary_2 = (y_pred_2 == y_true).astype(int)
    fraction_of_positives_2, mean_predicted_value_2 = calibration_curve(
        y_binary_2, max_proba_2, n_bins=n_bins, strategy="quantile"
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax1, ax2 = axes
    ax1.plot(
        mean_predicted_value_1,
        fraction_of_positives_1,
        "s-",
        label=model_name_1,
        color="#2E86AB",
        linewidth=2.5,
        markersize=8,
        alpha=0.9,
    )
    ax1.plot(
        mean_predicted_value_2,
        fraction_of_positives_2,
        "o-",
        label=model_name_2,
        color="#A23B72",
        linewidth=2.5,
        markersize=8,
        alpha=0.9,
    )
    ax1.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Идеально откалиброванная модель",
        linewidth=2,
        linestyle="--",
        alpha=0.7,
    )
    ax1.set_xlabel("Средняя предсказанная вероятность", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Доля правильных предсказаний", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Диаграмма надёжности (сравнение моделей)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax1.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax2.hist(
        max_proba_1,
        bins=n_bins,
        edgecolor="black",
        alpha=0.6,
        color="#2E86AB",
        label=model_name_1,
        density=True,
    )
    ax2.hist(
        max_proba_2,
        bins=n_bins,
        edgecolor="black",
        alpha=0.6,
        color="#A23B72",
        label=model_name_2,
        density=True,
    )
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.set_xlabel("Предсказанная вероятность", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Плотность", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Распределение максимальных вероятностей",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax2.grid(alpha=0.3, axis="y", linestyle="--")
    stats_text = f"\nСтатистика уверенности:\n\n{model_name_1}:\n  Среднее: {max_proba_1.mean():.4f}\n  Медиана: {np.median(max_proba_1):.4f}\n\n{model_name_2}:\n  Среднее: {max_proba_2.mean():.4f}\n  Медиана: {np.median(max_proba_2):.4f}\n    "
    ax2.text(
        0.98,
        0.98,
        stats_text,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_all_visualizations(
    results_dir: Path,
    models_dir: Path,
    data_dir: Path,
    label_mapping_path: Path,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    label_mapping = load_label_mapping(label_mapping_path)
    print("Создание визуализаций...")
    oof_lr = pd.read_csv(results_dir / "logistic_regression_oof_predictions.csv")
    oof_rf = pd.read_csv(results_dir / "random_forest_oof_predictions.csv")
    test_data = pd.read_csv(data_dir / "banking49_test.csv")
    import joblib

    vectorizer_lr = joblib.load(models_dir / "logistic_regression_vectorizer.pkl")
    model_lr = joblib.load(models_dir / "logistic_regression_model.pkl")
    vectorizer_rf = joblib.load(models_dir / "random_forest_vectorizer.pkl")
    model_rf = joblib.load(models_dir / "random_forest_model.pkl")
    X_test = test_data["text"]
    y_test = test_data["label"].values
    X_test_vec_lr = vectorizer_lr.transform(X_test)
    X_test_vec_rf = vectorizer_rf.transform(X_test)
    y_pred_test_lr = model_lr.predict(X_test_vec_lr)
    y_pred_test_rf = model_rf.predict(X_test_vec_rf)
    y_proba_test_lr = model_lr.predict_proba(X_test_vec_lr)
    y_proba_test_rf = model_rf.predict_proba(X_test_vec_rf)
    metrics_lr_oof = calculate_metrics(
        oof_lr["true_label"].values, oof_lr["oof_prediction"].values
    )
    metrics_rf_oof = calculate_metrics(
        oof_rf["true_label"].values, oof_rf["oof_prediction"].values
    )
    metrics_lr_test = calculate_metrics(y_test, y_pred_test_lr)
    metrics_rf_test = calculate_metrics(y_test, y_pred_test_rf)
    fold_accs_lr = [0.8407, 0.8515, 0.8641, 0.8543, 0.8415]
    fold_accs_rf = [0.6401, 0.6537, 0.6664, 0.6832, 0.6517]
    plot_metrics_comparison(
        metrics_lr_test, metrics_rf_test, output_dir / "metrics_comparison.png"
    )
    plot_cv_folds_comparison(
        fold_accs_lr, fold_accs_rf, output_dir / "cv_folds_comparison.png"
    )
    plot_confusion_matrix_top_classes(
        y_test,
        y_pred_test_lr,
        label_mapping,
        top_n=10,
        model_name="Logistic Regression",
        save_path=output_dir / "confusion_matrix_lr.png",
    )
    plot_confusion_matrix_top_classes(
        y_test,
        y_pred_test_rf,
        label_mapping,
        top_n=10,
        model_name="Random Forest",
        save_path=output_dir / "confusion_matrix_rf.png",
    )
    plot_class_distribution(
        test_data["label"].values,
        label_mapping,
        top_n=15,
        save_path=output_dir / "class_distribution.png",
    )
    n_classes = len(label_mapping)
    plot_roc_curve_multiclass(
        y_test,
        y_proba_test_lr,
        "Logistic Regression",
        save_path=output_dir / "roc_curve_lr.png",
        n_classes=n_classes,
    )
    plot_roc_curve_multiclass(
        y_test,
        y_proba_test_rf,
        "Random Forest",
        save_path=output_dir / "roc_curve_rf.png",
        n_classes=n_classes,
    )
    plot_precision_recall_curve_multiclass(
        y_test,
        y_proba_test_lr,
        "Logistic Regression",
        save_path=output_dir / "precision_recall_curve_lr.png",
        n_classes=n_classes,
    )
    plot_precision_recall_curve_multiclass(
        y_test,
        y_proba_test_rf,
        "Random Forest",
        save_path=output_dir / "precision_recall_curve_rf.png",
        n_classes=n_classes,
    )
    plot_reliability_diagram_comparison(
        y_test,
        y_proba_test_lr,
        "Logistic Regression",
        y_proba_test_rf,
        "Random Forest",
        save_path=output_dir / "reliability_diagram_comparison.png",
    )
    metrics_table = pd.DataFrame(
        {
            "Logistic Regression (OOF)": metrics_lr_oof,
            "Random Forest (OOF)": metrics_rf_oof,
            "Logistic Regression (Test)": metrics_lr_test,
            "Random Forest (Test)": metrics_rf_test,
        }
    )
    metrics_table.to_csv(output_dir / "metrics_table.csv")
    print(f"Визуализации сохранены в: {output_dir}")
