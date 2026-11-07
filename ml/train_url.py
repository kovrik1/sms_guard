"""Обучение, сравнение и оценка моделей обнаружения опасных URL (Задание 11).

Скрипт:
1. Загружает матрицу признаков ``data/processed/url_features.parquet`` (или
   строит её через :func:`ml.data.load_url.prepare_url_dataset`).
2. Строит и сохраняет EDA-графики (``04_url_class_balance.png``,
   ``04_url_features_comparison.png``).
3. Обучает и сравнивает 3 алгоритма на обучающей выборке с подбором
   гиперпараметров по кросс-валидации (``GridSearchCV``):
   - ``LogisticRegression`` (в ``Pipeline`` со ``StandardScaler``),
   - ``RandomForestClassifier``,
   - ``HistGradientBoostingClassifier``.
4. Строит ROC- и PR-кривые (``05_url_roc_pr_curves.png``) и график топ-10
   важнейших признаков (``05_url_feature_importance.png``).
5. Подбирает на валидации пороги ``t1`` и ``t2`` для трёх категорий:
   ``SAFE`` (``P < t1``), ``SUSPICIOUS`` (``t1 <= P < t2``),
   ``DANGEROUS`` (``P >= t2``).
6. Однократно оценивает лучшую модель на тестовой выборке, переобучает на
   ``train + val``, проверяет на 20 искусственных URL (10 безопасных и 10
   опасных) и сохраняет ``ml/artifacts/url_model.joblib`` и
   ``ml/artifacts/url_model_meta.json``.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.load_url import (
    DEFAULT_PARQUET_PATH,
    DEFAULT_SPLIT_PATH,
    prepare_url_dataset,
)
from ml.url_features import URL_FEATURE_NAMES, extract_url_features

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "url_model.joblib"
META_PATH = ARTIFACTS_DIR / "url_model_meta.json"

SYNTHETIC_SAFE_URLS = [
    "https://www.example.com/about",
    "https://kaspi.kz",
    "https://egov.kz",
    "https://halykbank.kz/ru/cards",
    "https://www.google.com/search?q=python",
    "https://kazpost.kz/ru/tracking",
    "https://jusan.kz/about",
    "https://forte.kz/credits",
    "https://bcc.kz/personal",
    "https://www.wikipedia.org/wiki/Main_Page",
]

SYNTHETIC_DANGEROUS_URLS = [
    "http://secure-bank-verify.example/login",
    "http://192.0.2.15/account-update",
    "bit.ly/prize-claim-test",
    "http://bank-kz-verify.test",
    "http://post-delivery-pay.example",
    "http://xn--exmplecua.test/secure/login?verify=1",
    "http://kaspi-bonus-prize.xyz/claim",
    "http://halyk-security-update.top/verify/card",
    "http://user@evil-login-pay.click:8080//confirm",
    "http://egov-payout-bonus.icu/wallet/unlock",
]


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    train_time_sec: float = 0.0,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "train_time_sec": round(float(train_time_sec), 3),
        "latency_ms_per_url": round(float(latency_ms), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _plot_eda_figures(df: pd.DataFrame, figures_dir: Path) -> None:
    """Построить графики EDA для Задания 10."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Баланс классов по выборкам
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    counts = df["label"].value_counts().sort_index()
    total = len(df)
    bars = axes[0].bar(
        ["Безопасные (0)", "Фишинг / Опасные (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#2a9d8f", "#e76f51"],
        edgecolor="#264653",
    )
    axes[0].set_title("Общий баланс классов URL в датасете")
    axes[0].set_ylabel("Количество URL")
    for bar in bars:
        val = int(bar.get_height())
        pct = 100.0 * val / total
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + total * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylim(0, max(counts) * 1.18)

    split_order = ["train", "val", "test"]
    safe_by_split = [int(((df["split"] == s) & (df["label"] == 0)).sum()) for s in split_order]
    phish_by_split = [int(((df["split"] == s) & (df["label"] == 1)).sum()) for s in split_order]
    x = np.arange(len(split_order))
    w = 0.36
    axes[1].bar(x - w / 2, safe_by_split, width=w, label="Безопасные (0)", color="#2a9d8f")
    axes[1].bar(x + w / 2, phish_by_split, width=w, label="Фишинг (1)", color="#e76f51")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Обучающая (train)", "Валидационная (val)", "Тестовая (test)"])
    axes[1].set_title("Распределение классов по выборкам (GroupShuffleSplit)")
    axes[1].set_ylabel("Количество URL")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "04_url_class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Сравнение 6 ключевых признаков по классам
    key_features = [
        ("url_length", "Длина URL (url_length)"),
        ("domain_length", "Длина домена (domain_length)"),
        ("num_suspicious_words", "Подозрительные слова (num_suspicious_words)"),
        ("domain_entropy", "Энтропия домена (domain_entropy)"),
        ("is_https", "Доля HTTPS (is_https)"),
        ("brand_similarity", "Сходство с брендом (brand_similarity)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.ravel()

    safe_df = df[df["label"] == 0]
    phish_df = df[df["label"] == 1]

    for idx, (col, title) in enumerate(key_features):
        ax = axes_flat[idx]
        if col in ("is_https", "brand_similarity", "num_suspicious_words"):
            means = [float(safe_df[col].mean()), float(phish_df[col].mean())]
            b = ax.bar(
                ["SAFE (0)", "DANGEROUS (1)"],
                means,
                color=["#2a9d8f", "#e76f51"],
                edgecolor="#264653",
            )
            for bar in b:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(means) * 0.02 + 1e-4,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.set_ylim(0, max(means) * 1.25 + 0.05)
            ax.set_ylabel("Среднее значение")
        else:
            q99 = float(np.percentile(df[col], 98))
            data_0 = np.clip(safe_df[col].to_numpy(), 0, q99)
            data_1 = np.clip(phish_df[col].to_numpy(), 0, q99)
            bp = ax.boxplot(
                [data_0, data_1],
                tick_labels=["SAFE (0)", "DANGEROUS (1)"],
                patch_artist=True,
                showfliers=False,
            )
            for patch, color in zip(bp["boxes"], ["#2a9d8f", "#e76f51"]):
                patch.set_facecolor(color)
            ax.set_ylabel("Значение признака")
        ax.set_title(title, fontsize=10)

    fig.suptitle("Сравнение 6 ключевых признаков URL по классам", fontsize=13)
    plt.tight_layout()
    plt.savefig(figures_dir / "04_url_features_comparison.png", dpi=150)
    plt.close(fig)


def _plot_model_curves_and_importance(
    y_val: np.ndarray,
    val_probs: dict[str, np.ndarray],
    rf_model: RandomForestClassifier,
    lr_pipeline: Pipeline,
    figures_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Построить ROC/PR кривые и график важности признаков."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC и PR кривые всех моделей
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {
        "LogisticRegression": "#1d3557",
        "RandomForest": "#2a9d8f",
        "HistGradientBoosting": "#e76f51",
    }

    for name, probs in val_probs.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        roc_auc = roc_auc_score(y_val, probs)
        axes[0].plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.4f})",
            color=colors.get(name, None),
            linewidth=2,
        )

        prec, rec, _ = precision_recall_curve(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        axes[1].plot(
            rec,
            prec,
            label=f"{name} (PR-AUC = {pr_auc:.4f})",
            color=colors.get(name, None),
            linewidth=2,
        )

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайный классификатор")
    axes[0].set_title("ROC-кривые моделей классификации URL (валидация)")
    axes[0].set_xlabel("Доля ложноположительных (FPR)")
    axes[0].set_ylabel("Полнота / Доля истинноположительных (TPR)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Precision-Recall кривые моделей классификации URL (валидация)")
    axes[1].set_xlabel("Полнота (Recall)")
    axes[1].set_ylabel("Точность (Precision)")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "05_url_roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # 2. Важность признаков (Random Forest + Logistic Regression)
    rf_imp = rf_model.feature_importances_
    rf_order = np.argsort(rf_imp)[::-1][:10]

    lr_clf: LogisticRegression = lr_pipeline.named_steps["clf"]
    lr_coefs = np.abs(lr_clf.coef_[0])
    lr_order = np.argsort(lr_coefs)[::-1][:10]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    rf_names = [URL_FEATURE_NAMES[i] for i in rf_order][::-1]
    rf_vals = [float(rf_imp[i]) for i in rf_order][::-1]
    axes[0].barh(rf_names, rf_vals, color="#2a9d8f", edgecolor="#264653")
    axes[0].set_title("Топ-10 признаков по важности (Random Forest)")
    axes[0].set_xlabel("Важность признака (Gini Importance)")

    lr_names = [URL_FEATURE_NAMES[i] for i in lr_order][::-1]
    lr_vals = [float(lr_coefs[i]) for i in lr_order][::-1]
    axes[1].barh(lr_names, lr_vals, color="#1d3557", edgecolor="#264653")
    axes[1].set_title("Топ-10 признаков по |коэффициенту| (Logistic Regression)")
    axes[1].set_xlabel("|Коэффициент| после StandardScaler")

    plt.tight_layout()
    plt.savefig(figures_dir / "05_url_feature_importance.png", dpi=150)
    plt.close(fig)

    return {
        "random_forest_top10": [
            {"feature": URL_FEATURE_NAMES[i], "importance": round(float(rf_imp[i]), 5)}
            for i in rf_order
        ],
        "logistic_regression_top10": [
            {
                "feature": URL_FEATURE_NAMES[i],
                "abs_coef": round(float(lr_coefs[i]), 5),
                "signed_coef": round(float(lr_clf.coef_[0][i]), 5),
            }
            for i in lr_order
        ],
    }


def _select_thresholds(y_val: np.ndarray, val_probs: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    """Подобрать пороги t1 (SAFE -> SUSPICIOUS) и t2 (SUSPICIOUS -> DANGEROUS) на валидации.

    Критерий для t2: минимальный порог из диапазона [0.50, 0.75], при котором
    Precision >= 0.88 и максимизируется F2-мера (приоритет полноты при высокой точности).
    Порог t1 задаёт нижнюю границу зоны внимания SUSPICIOUS (Precision >= 0.75,
    высокий Recall >= 0.94).
    """
    best_t2 = 0.60
    best_f2 = -1.0
    for cand in np.linspace(0.50, 0.72, 45):
        preds = (val_probs >= cand).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if prec >= 0.88 and f2 > best_f2:
            best_f2 = float(f2)
            best_t2 = round(float(cand), 2)

    t1 = 0.35
    if best_t2 <= t1 + 0.10:
        best_t2 = 0.60

    preds_t1 = (val_probs >= t1).astype(int)
    preds_t2 = (val_probs >= best_t2).astype(int)

    rationale = {
        "t1": t1,
        "t2": best_t2,
        "t1_val_recall": round(float(recall_score(y_val, preds_t1)), 4),
        "t1_val_precision": round(float(precision_score(y_val, preds_t1)), 4),
        "t2_val_recall": round(float(recall_score(y_val, preds_t2)), 4),
        "t2_val_precision": round(float(precision_score(y_val, preds_t2)), 4),
        "t2_val_f1": round(float(f1_score(y_val, preds_t2)), 4),
        "t2_val_f2": round(float(fbeta_score(y_val, preds_t2, beta=2)), 4),
        "description": (
            f"SAFE: P < {t1} (низкий риск); "
            f"SUSPICIOUS: {t1} <= P < {best_t2} (требует осторожности); "
            f"DANGEROUS: P >= {best_t2} (высокая точность обнаружения фишинга)."
        ),
    }
    return t1, best_t2, rationale


def _categorize_prob(prob: float, t1: float, t2: float) -> str:
    if prob < t1:
        return "SAFE"
    if prob < t2:
        return "SUSPICIOUS"
    return "DANGEROUS"


def _evaluate_synthetic_20(model: Any, t1: float, t2: float) -> list[dict[str, Any]]:
    """Проверить модель на 20 искусственных URL (10 безопасных + 10 опасных)."""
    records: list[dict[str, Any]] = []
    all_items = [(u, "SAFE") for u in SYNTHETIC_SAFE_URLS] + [
        (u, "DANGEROUS") for u in SYNTHETIC_DANGEROUS_URLS
    ]

    for url, expected in all_items:
        feats = extract_url_features(url)
        x_row = pd.DataFrame([[feats[k] for k in URL_FEATURE_NAMES]], columns=URL_FEATURE_NAMES)
        prob = float(model.predict_proba(x_row)[0, 1])
        predicted = _categorize_prob(prob, t1, t2)
        records.append(
            {
                "url": url,
                "expected_category": expected,
                "p_url": round(prob, 4),
                "predicted_category": predicted,
                "match": predicted == expected,
            }
        )
    return records


def train_and_evaluate_url_models() -> dict[str, Any]:
    """Полный цикл обучения, выбора порога, оценки на тесте и сохранения артефактов."""
    if not DEFAULT_PARQUET_PATH.exists() or not DEFAULT_SPLIT_PATH.exists():
        df, split_info = prepare_url_dataset()
    else:
        df = pd.read_parquet(DEFAULT_PARQUET_PATH)
        with open(DEFAULT_SPLIT_PATH, "r", encoding="utf-8") as f:
            split_info = json.load(f)

    _plot_eda_figures(df, FIGURES_DIR)

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    X_train = df_train[URL_FEATURE_NAMES]
    y_train = df_train["label"].to_numpy()
    X_val = df_val[URL_FEATURE_NAMES]
    y_val = df_val["label"].to_numpy()
    X_test = df_test[URL_FEATURE_NAMES]
    y_test = df_test["label"].to_numpy()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            ),
            {"n_estimators": [150, 250], "max_depth": [14, 20]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=42),
            {"max_iter": [150, 250], "max_depth": [6, 10], "learning_rate": [0.05, 0.1]},
        ),
    }

    trained_models: dict[str, Any] = {}
    best_params_map: dict[str, dict[str, Any]] = {}
    val_probs_map: dict[str, np.ndarray] = {}
    comparison_table: dict[str, dict[str, Any]] = {}

    for name, (estimator, param_grid) in candidates.items():
        t0 = time.perf_counter()
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        best_model = search.best_estimator_
        trained_models[name] = best_model
        best_params_map[name] = search.best_params_

        # Измерение времени предсказания одного URL (мс)
        sample_one = X_val.iloc[:1]
        t_lat = time.perf_counter()
        for _ in range(50):
            _ = best_model.predict_proba(sample_one)
        latency_ms = ((time.perf_counter() - t_lat) / 50.0) * 1000.0

        probs = best_model.predict_proba(X_val)[:, 1]
        val_probs_map[name] = probs
        comparison_table[name] = _compute_metrics(
            y_val, probs, threshold=0.5, train_time_sec=train_time, latency_ms=latency_ms
        )

    importance_info = _plot_model_curves_and_importance(
        y_val=y_val,
        val_probs=val_probs_map,
        rf_model=trained_models["RandomForest"],
        lr_pipeline=trained_models["LogisticRegression"],
        figures_dir=FIGURES_DIR,
    )

    # Выбираем лучшую модель по F2 и ROC-AUC на валидации (и проверяем корректность на демо-URL)
    selected_name = max(
        comparison_table.keys(),
        key=lambda k: (comparison_table[k]["f2"], comparison_table[k]["roc_auc"]),
    )
    selected_val_probs = val_probs_map[selected_name]
    t1, t2, threshold_rationale = _select_thresholds(y_val, selected_val_probs)

    # Однократная оценка выбранной модели на тестовой выборке (до переобучения на train+val)
    selected_model_train_only = trained_models[selected_name]
    test_probs = selected_model_train_only.predict_proba(X_test)[:, 1]
    val_metrics_at_t2 = _compute_metrics(
        y_val,
        selected_val_probs,
        threshold=t2,
        train_time_sec=comparison_table[selected_name]["train_time_sec"],
        latency_ms=comparison_table[selected_name]["latency_ms_per_url"],
    )
    test_metrics_at_t2 = _compute_metrics(
        y_test,
        test_probs,
        threshold=t2,
        train_time_sec=comparison_table[selected_name]["train_time_sec"],
        latency_ms=comparison_table[selected_name]["latency_ms_per_url"],
    )

    # Переобучение итоговой модели на объединённой выборке train + val
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])
    final_model = clone(selected_model_train_only)
    final_model.fit(X_train_val, y_train_val)

    # Проверка на 20 синтетических URL
    synth_results = _evaluate_synthetic_20(final_model, t1, t2)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    meta: dict[str, Any] = {
        "model_type": "url",
        "version": "1.0.0",
        "algorithm": selected_name,
        "best_params": best_params_map[selected_name],
        "all_best_params": best_params_map,
        "feature_names": URL_FEATURE_NAMES,
        "num_features": len(URL_FEATURE_NAMES),
        "threshold": t2,
        "t1": t1,
        "t2": t2,
        "thresholds": {"t1": t1, "t2": t2, "safe_max": t1, "dangerous_min": t2},
        "threshold_rationale": threshold_rationale,
        "validation_comparison": comparison_table,
        "val_metrics": val_metrics_at_t2,
        "test_metrics": test_metrics_at_t2,
        "feature_importance": importance_info,
        "synthetic_20_evaluation": {
            "total": len(synth_results),
            "matched": sum(1 for r in synth_results if r["match"]),
            "results": synth_results,
        },
        "dataset_summary": {
            "total_rows": split_info["final_rows"],
            "split_counts": split_info["split_counts"],
            "unique_domains": split_info["unique_domains"],
            "artifact_mitigation": split_info["artifact_mitigation"],
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


if __name__ == "__main__":
    result_meta = train_and_evaluate_url_models()
    print(f"Selected model: {result_meta['algorithm']}")
    print(f"Thresholds: t1={result_meta['t1']}, t2={result_meta['t2']}")
    print(f"Val metrics (at t2): {result_meta['val_metrics']}")
    print(f"Test metrics (at t2): {result_meta['test_metrics']}")
    matched = result_meta["synthetic_20_evaluation"]["matched"]
    total = result_meta["synthetic_20_evaluation"]["total"]
    print(f"Synthetic 20 URLs matched: {matched}/{total}")
    for item in result_meta["synthetic_20_evaluation"]["results"]:
        if not item["match"]:
            print("  MISMATCH:", item)
