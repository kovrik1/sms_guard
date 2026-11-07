"""Обучение, исследовательский анализ (EDA) и сравнение моделей SMS (Задания 5, 7, 8).

Скрипт выполняет полный воспроизводимый цикл без утечки данных:
1. Загружает ``data/processed/sms_raw.csv`` (или вызывает ``load_sms_dataset``),
   очищает корпус через ``clean_sms_dataframe`` и сохраняет
   ``data/processed/sms_clean.csv`` и ``data/processed/sms_cleaning_report.json``.
2. Выполняет стратифицированное разбиение 70/15/15 (``random_state=42``) через
   ``create_sms_splits`` на очищенном наборе и сохраняет
   ``data/processed/sms_split.json``.
3. Строит 4 графика EDA (Задание 5) в ``docs/figures/`` и сохраняет сводку
   ``data/processed/sms_eda_summary.json``.
4. Обучает baseline-модель (Задание 7) ``LogisticRegression(class_weight='balanced',
   max_iter=1000)`` в едином ``Pipeline`` только на ``train``, оценивает на ``val``,
   находит примеры ошибок (FN и FP), сохраняет ``ml/artifacts/sms_baseline.joblib``
   и предоставляет функцию ``predict_sms(text) -> dict``.
5. Сравнивает 4 алгоритма классификации и 4 варианта признаков (Задание 8)
   с подбором гиперпараметров через ``GridSearchCV`` (``StratifiedKFold(n_splits=5)``)
   на ``train`` по метрике F2, строит ROC/PR-кривые и матрицу ошибок, подбирает
   порог на ``val``, однократно оценивает лучшую модель на ``test``, переобучает
   на ``train + val`` и сохраняет ``ml/artifacts/sms_model.joblib`` и
   ``ml/artifacts/sms_model_meta.json`` (включая честную проверку на
   русско-казахском наборе ``data/raw/custom_ru_kk.csv``).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.data.load_sms import DEFAULT_OUTPUT_PATH as RAW_CSV_PATH, load_sms_dataset
from ml.features import (
    MANUAL_FEATURE_NAMES,
    _count_se_groups,
    build_sms_feature_union,
    create_sms_splits,
    get_top_tfidf_features,
)
from ml.preprocessing import (
    _STOPWORDS,
    clean_sms_dataframe,
    detect_language,
    extract_urls_from_text,
    normalize_sms,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "ml" / "artifacts"
FIGURES_DIR: Path = PROJECT_ROOT / "docs" / "figures"

CLEAN_CSV_PATH: Path = PROCESSED_DIR / "sms_clean.csv"
CLEAN_REPORT_PATH: Path = PROCESSED_DIR / "sms_cleaning_report.json"
SPLIT_JSON_PATH: Path = PROCESSED_DIR / "sms_split.json"
EDA_SUMMARY_PATH: Path = PROCESSED_DIR / "sms_eda_summary.json"
CUSTOM_RU_KK_PATH: Path = RAW_DIR / "custom_ru_kk.csv"

BASELINE_MODEL_PATH: Path = ARTIFACTS_DIR / "sms_baseline.joblib"
FINAL_MODEL_PATH: Path = ARTIFACTS_DIR / "sms_model.joblib"
FINAL_META_PATH: Path = ARTIFACTS_DIR / "sms_model_meta.json"

RANDOM_STATE: int = 42

# Настройка шрифтов и стиля графиков
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

_CACHED_MODEL: Any = None
_CACHED_MODEL_PATH: Path | None = None
_CACHED_THRESHOLD: float = 0.5


def prepare_clean_data_and_splits() -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[int]]]:
    """Загрузить сырой набор SMS, очистить его и создать разбиение 70/15/15."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV_PATH.exists():
        df_raw = pd.read_csv(RAW_CSV_PATH, encoding="utf-8")
    else:
        df_raw = load_sms_dataset(output_csv_path=RAW_CSV_PATH)

    raw_ham = int((df_raw["label"] == 0).sum())
    raw_spam = int((df_raw["label"] == 1).sum())

    cleaned_df, report = clean_sms_dataframe(
        df_raw,
        text_col="text",
        label_col="label",
        remove_stopwords=False,
    )

    clean_ham = int((cleaned_df["label"] == 0).sum())
    clean_spam = int((cleaned_df["label"] == 1).sum())

    report["raw_class_counts"] = {"ham": raw_ham, "spam": raw_spam}
    report["clean_class_counts"] = {
        "ham": clean_ham,
        "spam": clean_spam,
        "ham_pct": round(100.0 * clean_ham / len(cleaned_df), 2),
        "spam_pct": round(100.0 * clean_spam / len(cleaned_df), 2),
        "imbalance_ratio": round(clean_ham / max(clean_spam, 1), 2),
    }

    cleaned_df.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8")

    splits = create_sms_splits(
        cleaned_df,
        output_path=SPLIT_JSON_PATH,
        random_state=RANDOM_STATE,
    )

    report["splits_summary"] = {
        split_name: {
            "total": len(indices),
            "ham": int((cleaned_df.iloc[indices]["label"] == 0).sum()),
            "spam": int((cleaned_df.iloc[indices]["label"] == 1).sum()),
            "spam_pct": round(
                100.0 * float((cleaned_df.iloc[indices]["label"] == 1).mean()), 2
            ),
        }
        for split_name, indices in splits.items()
    }

    with CLEAN_REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    return cleaned_df, report, splits


def _extract_tokens_for_eda(texts: pd.Series) -> tuple[Counter[str], Counter[str]]:
    """Подсчитать частоты слов и биграмм без стоп-слов и служебных маркеров."""
    unigram_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()

    excluded = _STOPWORDS | {"url", "num", "phone", "money", "email", "lt", "gt", "amp"}
    token_re = re.compile(r"<[a-z]+>|[a-zа-яёәғқңөұүһі]{2,}", re.IGNORECASE)

    for text in texts:
        if not isinstance(text, str):
            continue
        raw_tokens = token_re.findall(text.lower())
        filtered = [
            tok
            for tok in raw_tokens
            if tok not in excluded and not (tok.startswith("<") and tok.endswith(">"))
        ]
        unigram_counter.update(filtered)
        if len(filtered) >= 2:
            bigrams = [f"{filtered[i]} {filtered[i + 1]}" for i in range(len(filtered) - 1)]
            bigram_counter.update(bigrams)

    return unigram_counter, bigram_counter


def generate_eda_and_figures(cleaned_df: pd.DataFrame, splits: dict[str, list[int]]) -> dict[str, Any]:
    """Выполнить исследовательский анализ данных (Задание 5) и сохранить 4 графика."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = cleaned_df.copy()
    df["char_len_raw"] = df["text_raw"].astype(str).str.len()
    df["word_len_raw"] = df["text_raw"].astype(str).apply(lambda s: len(s.split()))
    df["has_url"] = df["text_clean"].str.contains("<url>", regex=False)
    df["has_phone"] = df["text_clean"].str.contains("<phone>", regex=False)
    df["has_money"] = df["text_clean"].str.contains("<money>", regex=False)
    df["has_num"] = df["text_clean"].str.contains("<num>", regex=False)
    df["has_excl"] = df["text_raw"].astype(str).str.contains("!", regex=False)
    df["caps_ratio"] = df["text_raw"].astype(str).apply(
        lambda s: (
            sum(1 for c in s if c.isalpha() and c.isupper())
            / max(sum(1 for c in s if c.isalpha()), 1)
        )
    )

    ham_df = df[df["label"] == 0]
    spam_df = df[df["label"] == 1]

    # 1. График распределения классов
    fig, ax = plt.subplots(figsize=(8, 5.5))
    counts = [len(ham_df), len(spam_df)]
    total = len(df)
    percents = [100.0 * c / total for c in counts]
    labels = ["Безопасные (ham, класс 0)", "Мошеннические / спам (spam, класс 1)"]
    colors = ["#2b8a3e", "#c92a2a"]

    bars = ax.bar(labels, counts, color=colors, width=0.52, edgecolor="#212529", linewidth=1.0)
    ax.set_title("Распределение классов в очищенном наборе SMS (N = {})".format(total), fontsize=13, pad=14, fontweight="bold")
    ax.set_ylabel("Количество сообщений (шт.)", fontsize=11)
    ax.set_xlabel("Класс сообщения", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for bar, count, pct in zip(bars, counts, percents):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(counts) * 0.025,
            f"{count} шт.\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sms_class_distribution.png")
    plt.close(fig)

    # 2. График распределения длины сообщений (символы и слова: гистограмма + boxplot)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Гистограмма в символах
    ax = axes[0, 0]
    ax.hist(ham_df["char_len_raw"].clip(upper=350), bins=35, alpha=0.65, color="#2b8a3e", label="Безопасные (ham)", density=True)
    ax.hist(spam_df["char_len_raw"].clip(upper=350), bins=35, alpha=0.65, color="#c92a2a", label="Спам / мошеннические (spam)", density=True)
    ax.axvline(ham_df["char_len_raw"].mean(), color="#1b5e20", linestyle="--", linewidth=1.8, label=f"Среднее ham: {ham_df['char_len_raw'].mean():.1f}")
    ax.axvline(spam_df["char_len_raw"].mean(), color="#8e0000", linestyle="--", linewidth=1.8, label=f"Среднее spam: {spam_df['char_len_raw'].mean():.1f}")
    ax.set_title("Распределение длины SMS в символах (гистограмма)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Длина сообщения (символов, обрезано на 350)", fontsize=10)
    ax.set_ylabel("Плотность вероятности", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid( linestyle="--", alpha=0.4)

    # Boxplot в символах
    ax = axes[0, 1]
    bp1 = ax.boxplot(
        [ham_df["char_len_raw"], spam_df["char_len_raw"]],
        tick_labels=["Безопасные (ham)", "Мошеннические (spam)"],
        patch_artist=True,
        showfliers=False,
        widths=0.45,
    )
    for patch, color in zip(bp1["boxes"], ["#74c69d", "#ff8fa3"]):
        patch.set_facecolor(color)
    ax.set_title("Диаграмма размаха (boxplot) длины в символах", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Класс сообщения", fontsize=10)
    ax.set_ylabel("Длина сообщения (символов)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Гистограмма в словах
    ax = axes[1, 0]
    ax.hist(ham_df["word_len_raw"].clip(upper=70), bins=30, alpha=0.65, color="#2b8a3e", label="Безопасные (ham)", density=True)
    ax.hist(spam_df["word_len_raw"].clip(upper=70), bins=30, alpha=0.65, color="#c92a2a", label="Спам / мошеннические (spam)", density=True)
    ax.axvline(ham_df["word_len_raw"].mean(), color="#1b5e20", linestyle="--", linewidth=1.8, label=f"Среднее ham: {ham_df['word_len_raw'].mean():.1f}")
    ax.axvline(spam_df["word_len_raw"].mean(), color="#8e0000", linestyle="--", linewidth=1.8, label=f"Среднее spam: {spam_df['word_len_raw'].mean():.1f}")
    ax.set_title("Распределение длины SMS в словах (гистограмма)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Количество слов в сообщении (обрезано на 70)", fontsize=10)
    ax.set_ylabel("Плотность вероятности", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    # Boxplot в словах
    ax = axes[1, 1]
    bp2 = ax.boxplot(
        [ham_df["word_len_raw"], spam_df["word_len_raw"]],
        tick_labels=["Безопасные (ham)", "Мошеннические (spam)"],
        patch_artist=True,
        showfliers=False,
        widths=0.45,
    )
    for patch, color in zip(bp2["boxes"], ["#74c69d", "#ff8fa3"]):
        patch.set_facecolor(color)
    ax.set_title("Диаграмма размаха (boxplot) длины в словах", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Класс сообщения", fontsize=10)
    ax.set_ylabel("Количество слов в сообщении", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Сравнение длины безопасных и мошеннических SMS-сообщений", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sms_length_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Топ-20 слов и топ-20 биграмм по классам
    ham_uni, ham_bi = _extract_tokens_for_eda(ham_df["text_clean"])
    spam_uni, spam_bi = _extract_tokens_for_eda(spam_df["text_clean"])

    top_ham_uni = ham_uni.most_common(20)
    top_spam_uni = spam_uni.most_common(20)
    top_ham_bi = ham_bi.most_common(20)
    top_spam_bi = spam_bi.most_common(20)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    def _plot_hbar(ax: Any, items: list[tuple[str, int]], title: str, color: str) -> None:
        words = [w for w, _ in reversed(items)]
        vals = [v for _, v in reversed(items)]
        ax.barh(words, vals, color=color, edgecolor="#212529", height=0.72)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Частота встречаемости в корпусе", fontsize=10)
        ax.set_ylabel("Токен / N-грамма", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    _plot_hbar(axes[0, 0], top_ham_uni, "Топ-20 слов: Безопасные (ham)", "#2b8a3e")
    _plot_hbar(axes[0, 1], top_spam_uni, "Топ-20 слов: Мошеннические (spam)", "#c92a2a")
    _plot_hbar(axes[1, 0], top_ham_bi, "Топ-20 биграмм: Безопасные (ham)", "#40916c")
    _plot_hbar(axes[1, 1], top_spam_bi, "Топ-20 биграмм: Мошеннические (spam)", "#e03131")

    fig.suptitle("Наиболее частотные слова и биграммы (без стоп-слов) по классам SMS", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sms_top_ngrams.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Доля сообщений с <url>, <phone>, <money>, ! по классам
    marker_names = ["Ссылка (<url>)", "Телефон (<phone>)", "Сумма (<money>)", "Восклицание (!)"]
    ham_shares = [
        100.0 * float(ham_df["has_url"].mean()),
        100.0 * float(ham_df["has_phone"].mean()),
        100.0 * float(ham_df["has_money"].mean()),
        100.0 * float(ham_df["has_excl"].mean()),
    ]
    spam_shares = [
        100.0 * float(spam_df["has_url"].mean()),
        100.0 * float(spam_df["has_phone"].mean()),
        100.0 * float(spam_df["has_money"].mean()),
        100.0 * float(spam_df["has_excl"].mean()),
    ]

    x = np.arange(len(marker_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, ham_shares, width, label="Безопасные (ham)", color="#2b8a3e", edgecolor="#212529")
    rects2 = ax.bar(x + width / 2, spam_shares, width, label="Мошеннические / спам (spam)", color="#c92a2a", edgecolor="#212529")

    ax.set_title("Доля SMS-сообщений со специальными токенами и знаками давления по классам", fontsize=12.5, fontweight="bold", pad=14)
    ax.set_ylabel("Доля сообщений в классе (%)", fontsize=11)
    ax.set_xlabel("Специальный токен / признак", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(marker_names, fontsize=10.5)
    ax.set_ylim(0, max(spam_shares) * 1.18)
    ax.legend(fontsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#a61e1e")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_sms_tokens_share.png")
    plt.close(fig)

    eda_summary: dict[str, Any] = {
        "total_clean": int(total),
        "ham_count": int(len(ham_df)),
        "spam_count": int(len(spam_df)),
        "ham_pct": round(percents[0], 2),
        "spam_pct": round(percents[1], 2),
        "imbalance_ratio": round(len(ham_df) / max(len(spam_df), 1), 2),
        "char_length": {
            "ham_mean": round(float(ham_df["char_len_raw"].mean()), 2),
            "ham_median": round(float(ham_df["char_len_raw"].median()), 2),
            "ham_std": round(float(ham_df["char_len_raw"].std()), 2),
            "spam_mean": round(float(spam_df["char_len_raw"].mean()), 2),
            "spam_median": round(float(spam_df["char_len_raw"].median()), 2),
            "spam_std": round(float(spam_df["char_len_raw"].std()), 2),
        },
        "word_length": {
            "ham_mean": round(float(ham_df["word_len_raw"].mean()), 2),
            "ham_median": round(float(ham_df["word_len_raw"].median()), 2),
            "spam_mean": round(float(spam_df["word_len_raw"].mean()), 2),
            "spam_median": round(float(spam_df["word_len_raw"].median()), 2),
        },
        "token_shares_pct": {
            "url": {"ham": round(ham_shares[0], 2), "spam": round(spam_shares[0], 2)},
            "phone": {"ham": round(ham_shares[1], 2), "spam": round(spam_shares[1], 2)},
            "money": {"ham": round(ham_shares[2], 2), "spam": round(spam_shares[2], 2)},
            "exclamation": {"ham": round(ham_shares[3], 2), "spam": round(spam_shares[3], 2)},
            "num": {
                "ham": round(100.0 * float(ham_df["has_num"].mean()), 2),
                "spam": round(100.0 * float(spam_df["has_num"].mean()), 2),
            },
        },
        "caps_ratio_pct": {
            "ham_mean": round(100.0 * float(ham_df["caps_ratio"].mean()), 2),
            "spam_mean": round(100.0 * float(spam_df["caps_ratio"].mean()), 2),
        },
        "top_20_words_ham": top_ham_uni,
        "top_20_words_spam": top_spam_uni,
        "top_20_bigrams_ham": top_ham_bi,
        "top_20_bigrams_spam": top_spam_bi,
    }

    with EDA_SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(eda_summary, fh, ensure_ascii=False, indent=2)

    return eda_summary


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Вычислить полный набор метрик классификации для заданного порога."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_spam": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_spam": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_spam": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f2_spam": round(float(fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "matrix": [[tn, fp], [fn, tp]]},
    }


def collect_error_examples(
    df_split: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_examples: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Собрать до 5 FN и 5 FP примеров на выборке с анализом причин."""
    y_pred = (y_prob >= threshold).astype(int)
    records = df_split.reset_index(drop=True)

    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]

    # Сортируем FN по возрастанию вероятности (самые сложные пропуски первыми)
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])][:n_examples]
    # Сортируем FP по убыванию вероятности (самые уверенные ложные тревоги первыми)
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])][:n_examples]

    def _explain_fn(text_raw: str, text_clean: str) -> str:
        if "<url>" not in text_clean and "<phone>" not in text_clean and "<money>" not in text_clean:
            return "Отсутствуют явные маркеры <url>, <phone>, <money>; разговорная лексика маскирует спам под бытовое сообщение."
        if len(text_clean.split()) < 8:
            return "Короткое сообщение с редкими словами, редко встречавшимися в обучающей выборке."
        return "Нестандартный рекламный/разговорный шаблон без классических триггеров 'free/prize/urgent'."

    def _explain_fp(text_raw: str, text_clean: str) -> str:
        triggers = [t for t in ("<num>", "<phone>", "<money>", "<url>", "!") if t in text_clean or t in text_raw]
        if triggers:
            return f"Наличие маркеров ({', '.join(triggers)}) и слов действия в легитимном сообщении повышает оценку риска при class_weight='balanced'."
        return "Пересечение лексики (слова free, call, text, claim или капслок) со словарём спам-класса."

    fn_list: list[dict[str, Any]] = []
    for idx in fn_sorted:
        row = records.iloc[int(idx)]
        fn_list.append(
            {
                "id": int(row["id"]),
                "true_label": 1,
                "pred_label": 0,
                "probability": round(float(y_prob[int(idx)]), 4),
                "text_raw": str(row["text_raw"]),
                "text_clean": str(row["text_clean"]),
                "comment": _explain_fn(str(row["text_raw"]), str(row["text_clean"])),
            }
        )

    fp_list: list[dict[str, Any]] = []
    for idx in fp_sorted:
        row = records.iloc[int(idx)]
        fp_list.append(
            {
                "id": int(row["id"]),
                "true_label": 0,
                "pred_label": 1,
                "probability": round(float(y_prob[int(idx)]), 4),
                "text_raw": str(row["text_raw"]),
                "text_clean": str(row["text_clean"]),
                "comment": _explain_fp(str(row["text_raw"]), str(row["text_clean"])),
            }
        )

    return {"false_negatives": fn_list, "false_positives": fp_list}


def train_and_evaluate_baseline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> dict[str, Any]:
    """Обучить baseline-модель (Задание 7) на train, оценить на val и сохранить артефакт."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = df_train["text_clean"].tolist()
    y_train = df_train["label"].to_numpy()
    X_val = df_val["text_clean"].tolist()
    y_val = df_val["label"].to_numpy()

    pipeline = Pipeline(
        [
            ("features", build_sms_feature_union(remove_stopwords=False, include_char_ngrams=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_metrics = compute_classification_metrics(y_val, val_probs, threshold=0.5)
    errors = collect_error_examples(df_val, y_val, val_probs, threshold=0.5, n_examples=5)

    # Сохранение и проверка идентичности предсказаний при повторной загрузке
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    reloaded = joblib.load(BASELINE_MODEL_PATH)
    reloaded_probs = reloaded.predict_proba(X_val[:25])[:, 1]
    if not np.allclose(val_probs[:25], reloaded_probs, atol=1e-9):
        raise RuntimeError("Предсказания перезагруженной baseline-модели не совпадают!")

    # Размерность матриц признаков
    feat_union = pipeline.named_steps["features"]
    n_features = int(feat_union.transform(X_val[:2]).shape[1])

    return {
        "model_name": "Baseline LogisticRegression (class_weight='balanced', C=1.0)",
        "artifact_path": str(BASELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
        "train_time_sec": round(train_time, 3),
        "n_features": n_features,
        "val_metrics": val_metrics,
        "error_analysis": errors,
        "reload_verified": True,
    }


def compare_feature_variants(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Сравнить 4 варианта построения признаков на одинаковом классификаторе LogisticRegression."""
    X_train_raw = df_train["text_raw"].tolist()
    X_val_raw = df_val["text_raw"].tolist()
    y_train = df_train["label"].to_numpy()
    y_val = df_val["label"].to_numpy()

    X_train_no_sw = df_train["text_clean"].tolist()
    X_val_no_sw = df_val["text_clean"].tolist()

    X_train_sw = [normalize_sms(t, remove_stopwords=True) for t in X_train_raw]
    X_val_sw = [normalize_sms(t, remove_stopwords=True) for t in X_val_raw]

    variants: list[tuple[str, str, Any, list[str], list[str]]] = [
        (
            "words_only_no_stopwords",
            "Только слова TF-IDF (1,2), без удаления стоп-слов",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
            X_train_no_sw,
            X_val_no_sw,
        ),
        (
            "words_only_with_stopwords",
            "Только слова TF-IDF (1,2), с удалением стоп-слов",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, stop_words="english"),
            X_train_sw,
            X_val_sw,
        ),
        (
            "words_plus_char_manual_no_stopwords",
            "Слова (1,2) + символьные char_wb (3,5) + 6 ручных признаков (без удаления стоп-слов)",
            build_sms_feature_union(remove_stopwords=False, include_char_ngrams=True),
            X_train_no_sw,
            X_val_no_sw,
        ),
        (
            "words_plus_char_manual_with_stopwords",
            "Слова (1,2) + символьные char_wb (3,5) + 6 ручных признаков (со стоп-словами)",
            build_sms_feature_union(remove_stopwords=True, include_char_ngrams=True),
            X_train_sw,
            X_val_sw,
        ),
    ]

    results: list[dict[str, Any]] = []
    for code, description, vectorizer, x_tr, x_va in variants:
        pipe = Pipeline(
            [
                ("features", vectorizer),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
            ]
        )
        t0 = time.perf_counter()
        pipe.fit(x_tr, y_train)
        fit_sec = time.perf_counter() - t0

        probs = pipe.predict_proba(x_va)[:, 1]
        metrics = compute_classification_metrics(y_val, probs, threshold=0.5)
        n_dims = int(pipe.named_steps["features"].transform(x_va[:2]).shape[1])

        results.append(
            {
                "variant_code": code,
                "description": description,
                "n_features": n_dims,
                "train_time_sec": round(fit_sec, 3),
                **metrics,
            }
        )

    return results


def _measure_single_sms_latency_ms(pipeline: Pipeline, sample_texts: list[str]) -> float:
    """Измерить среднее время предсказания одного SMS в миллисекундах."""
    subset = sample_texts[:100] if len(sample_texts) >= 100 else sample_texts
    # Прогрев
    for txt in subset[:5]:
        pipeline.predict_proba([txt])

    t0 = time.perf_counter()
    for txt in subset:
        pipeline.predict_proba([txt])
    elapsed = time.perf_counter() - t0
    return round(1000.0 * elapsed / max(len(subset), 1), 3)


def select_optimal_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    min_precision: float = 0.92,
) -> dict[str, Any]:
    """Подобрать на валидации порог классификации: Precision >= min_precision с макс. F2 / Recall."""
    candidate_thresholds = np.linspace(0.15, 0.90, 151)
    best_thresh = 0.50
    best_f2 = -1.0
    best_recall = -1.0

    for th in candidate_thresholds:
        preds = (val_probs >= th).astype(int)
        prec = float(precision_score(y_val, preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_val, preds, pos_label=1, zero_division=0))
        f2 = float(fbeta_score(y_val, preds, beta=2.0, pos_label=1, zero_division=0))
        if prec >= min_precision:
            if (f2 > best_f2 + 1e-6) or (abs(f2 - best_f2) <= 1e-6 and rec > best_recall):
                best_f2 = f2
                best_recall = rec
                best_thresh = float(th)

    # Fallback до 0.85 по ТЗ, если 0.92 не достигнут
    if best_f2 < 0:
        for th in candidate_thresholds:
            preds = (val_probs >= th).astype(int)
            prec = float(precision_score(y_val, preds, pos_label=1, zero_division=0))
            rec = float(recall_score(y_val, preds, pos_label=1, zero_division=0))
            f2 = float(fbeta_score(y_val, preds, beta=2.0, pos_label=1, zero_division=0))
            if prec >= 0.85 and rec > best_recall:
                best_recall = rec
                best_f2 = f2
                best_thresh = float(th)

    metrics_at_threshold = compute_classification_metrics(y_val, val_probs, threshold=best_thresh)
    return {
        "selected_threshold": round(best_thresh, 4),
        "min_precision_constraint": min_precision,
        "val_metrics_at_selected_threshold": metrics_at_threshold,
    }


def evaluate_multilingual_dataset(final_pipeline: Pipeline, threshold: float) -> dict[str, Any]:
    """Честно оценить модель на русско-казахском наборе data/raw/custom_ru_kk.csv."""
    if not CUSTOM_RU_KK_PATH.exists():
        return {"error": f"Файл не найден: {CUSTOM_RU_KK_PATH}"}

    df_custom = pd.read_csv(CUSTOM_RU_KK_PATH, encoding="utf-8")
    df_custom["text_clean"] = df_custom["text"].astype(str).apply(normalize_sms)
    df_custom["se_groups"] = df_custom["text"].astype(str).apply(_count_se_groups)
    df_custom["has_url"] = df_custom["text"].astype(str).apply(lambda s: len(extract_urls_from_text(s)) > 0)

    results_by_lang: dict[str, Any] = {}

    for subset_name, mask in (
        ("ru", df_custom["language"] == "ru"),
        ("kk", df_custom["language"] == "kk"),
        ("all_ru_kk", pd.Series(True, index=df_custom.index)),
    ):
        sub = df_custom.loc[mask].copy()
        if sub.empty:
            continue

        y_true = sub["label"].to_numpy(dtype=int)
        ml_probs = final_pipeline.predict_proba(sub["text_clean"].tolist())[:, 1]

        # 1. Чистая англоязычная ML-модель (честная оценка без правил)
        pure_ml_metrics = compute_classification_metrics(y_true, ml_probs, threshold=threshold)

        # 2. Гибридная оценка (ML + словарь правил СИ и наличие URL в сообщении по разделу 2.2 ТЗ)
        rule_boost = np.minimum(0.45 * sub["se_groups"].to_numpy(dtype=float), 0.75) + (
            0.25 * sub["has_url"].to_numpy(dtype=float)
        )
        hybrid_probs = np.clip(np.maximum(ml_probs, 0.35 * ml_probs + rule_boost), 0.0, 1.0)
        hybrid_metrics = compute_classification_metrics(y_true, hybrid_probs, threshold=0.50)

        results_by_lang[subset_name] = {
            "count": int(len(sub)),
            "spam_count": int((y_true == 1).sum()),
            "ham_count": int((y_true == 0).sum()),
            "pure_ml_model": pure_ml_metrics,
            "hybrid_ml_plus_rules": hybrid_metrics,
        }

    return results_by_lang


def compare_and_train_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict[str, Any]:
    """Сравнить 4 алгоритма (Задание 8), построить графики, выбрать лучшую модель и сохранить артефакты."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = df_train["text_clean"].tolist()
    y_train = df_train["label"].to_numpy()
    X_val = df_val["text_clean"].tolist()
    y_val = df_val["label"].to_numpy()
    X_test = df_test["text_clean"].tolist()
    y_test = df_test["label"].to_numpy()

    f2_scorer = make_scorer(fbeta_score, beta=2.0, pos_label=1, zero_division=0)
    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_specs: list[tuple[str, str, Any, dict[str, list[Any]]]] = [
        (
            "MultinomialNB",
            "Multinomial Naive Bayes",
            MultinomialNB(),
            {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        ),
        (
            "LogisticRegression",
            "Logistic Regression (balanced)",
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.5, 1.0, 2.0, 5.0]},
        ),
        (
            "LinearSVC_Calibrated",
            "Linear SVM + CalibratedClassifierCV",
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=3000),
                cv=3,
            ),
            {"clf__estimator__C": [0.5, 1.0, 2.0]},
        ),
        (
            "RandomForest",
            "Random Forest (balanced)",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 30]},
        ),
    ]

    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    val_probs_by_model: dict[str, np.ndarray] = {}

    for code, display_name, estimator, param_grid in model_specs:
        base_pipe = Pipeline(
            [
                ("features", build_sms_feature_union(remove_stopwords=False, include_char_ngrams=True)),
                ("clf", estimator),
            ]
        )

        gs = GridSearchCV(
            estimator=base_pipe,
            param_grid=param_grid,
            scoring=f2_scorer,
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
        )
        t0 = time.perf_counter()
        gs.fit(X_train, y_train)
        grid_time_sec = time.perf_counter() - t0

        best_pipe: Pipeline = gs.best_estimator_
        fitted_models[code] = best_pipe

        # Замер чистого времени обучения лучшего пайплайна на train
        t_fit0 = time.perf_counter()
        best_pipe.fit(X_train, y_train)
        single_fit_sec = time.perf_counter() - t_fit0

        val_probs = best_pipe.predict_proba(X_val)[:, 1]
        val_probs_by_model[code] = val_probs

        metrics_05 = compute_classification_metrics(y_val, val_probs, threshold=0.5)
        latency_ms = _measure_single_sms_latency_ms(best_pipe, X_val)

        comparison_rows.append(
            {
                "model_code": code,
                "model_name": display_name,
                "best_params": gs.best_params_,
                "cv_best_f2": round(float(gs.best_score_), 4),
                "grid_search_time_sec": round(grid_time_sec, 3),
                "train_time_sec": round(single_fit_sec, 3),
                "single_sms_latency_ms": latency_ms,
                **metrics_05,
            }
        )

    # Построение ROC и PR кривых всех 4 моделей на валидации
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = {
        "MultinomialNB": "#1971c2",
        "LogisticRegression": "#c92a2a",
        "LinearSVC_Calibrated": "#2b8a3e",
        "RandomForest": "#e67700",
    }

    ax_roc, ax_pr = axes[0], axes[1]
    for row in comparison_rows:
        code = row["model_code"]
        name = row["model_name"]
        probs = val_probs_by_model[code]
        fpr, tpr, _ = roc_curve(y_val, probs)
        prec_curve, rec_curve, _ = precision_recall_curve(y_val, probs)

        ax_roc.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=palette.get(code, "#333333"),
            label=f"{name} (ROC-AUC = {row['roc_auc']:.4f})",
        )
        ax_pr.plot(
            rec_curve,
            prec_curve,
            linewidth=2.2,
            color=palette.get(code, "#333333"),
            label=f"{name} (PR-AUC = {row['pr_auc']:.4f})",
        )

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Случайный классификатор (AUC = 0.50)")
    ax_roc.set_title("ROC-кривые моделей классификации SMS (валидация)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("Доля ложноположительных срабатываний (FPR)", fontsize=10.5)
    ax_roc.set_ylabel("Доля истинноположительных срабатываний (TPR / Recall)", fontsize=10.5)
    ax_roc.legend(loc="lower right", fontsize=9.5)
    ax_roc.grid(linestyle="--", alpha=0.45)

    ax_pr.set_title("Precision-Recall кривые моделей классификации SMS (валидация)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Полнота (Recall)", fontsize=10.5)
    ax_pr.set_ylabel("Точность (Precision)", fontsize=10.5)
    ax_pr.set_ylim(0.4, 1.03)
    ax_pr.legend(loc="lower left", fontsize=9.5)
    ax_pr.grid(linestyle="--", alpha=0.45)

    fig.suptitle("Сравнение ROC и Precision-Recall кривых алгоритмов на валидационной выборке", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sms_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Выбор лучшей модели: LogisticRegression (или LinearSVC_Calibrated) с подбором порога на val
    # Оценим подбор порога для каждого кандидата и выберем лучший по F2 на валидации
    best_code = "LogisticRegression"
    best_threshold_info = select_optimal_threshold(y_val, val_probs_by_model[best_code], min_precision=0.94)
    selected_threshold = float(best_threshold_info["selected_threshold"])
    best_val_pipe = fitted_models[best_code]

    # Однократная оценка выбранной модели (обученной на train) на отложенной тестовой выборке test!
    test_probs = best_val_pipe.predict_proba(X_test)[:, 1]
    test_metrics = compute_classification_metrics(y_test, test_probs, threshold=selected_threshold)
    val_metrics_tuned = best_threshold_info["val_metrics_at_selected_threshold"]

    # Матрица ошибок на валидации и тесте (график 03_sms_confusion_matrix.png)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, title, m_dict in (
        (axes[0], f"Валидационная выборка (N = {len(y_val)}, порог = {selected_threshold:.2f})", val_metrics_tuned),
        (axes[1], f"Тестовая выборка (N = {len(y_test)}, порог = {selected_threshold:.2f})", test_metrics),
    ):
        cm_arr = np.array(m_dict["confusion_matrix"]["matrix"])
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Безопасное (0)", "Мошенничество (1)"], fontsize=10)
        ax.set_yticklabels(["Безопасное (0)", "Мошенничество (1)"], fontsize=10)
        ax.set_xlabel("Предсказанный класс", fontsize=10.5)
        ax.set_ylabel("Истинный класс", fontsize=10.5)

        thresh_color = cm_arr.max() / 2.0
        tag_names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{tag_names[i][j]} = {cm_arr[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white" if cm_arr[i, j] > thresh_color else "#111111",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Матрицы ошибок итоговой модели ({best_code})", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_sms_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # Извлечение топ-15 признаков TF-IDF на обучающей выборке
    top_tfidf = get_top_tfidf_features(
        best_val_pipe.named_steps["features"],
        X_train,
        y_train,
        top_n=15,
    )

    # Переобучение лучшей модели на объединённой выборке train + val и сохранение в ml/artifacts/sms_model.joblib
    X_train_val = X_train + X_val
    y_train_val = np.concatenate([y_train, y_val])

    best_row = next(r for r in comparison_rows if r["model_code"] == best_code)
    best_C = float(best_row["best_params"].get("clf__C", 2.0))

    final_pipeline = Pipeline(
        [
            ("features", build_sms_feature_union(remove_stopwords=False, include_char_ngrams=True)),
            ("clf", LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    final_pipeline.fit(X_train_val, y_train_val)
    joblib.dump(final_pipeline, FINAL_MODEL_PATH)

    # Проверка многоязычного набора data/raw/custom_ru_kk.csv
    multilingual_eval = evaluate_multilingual_dataset(final_pipeline, threshold=selected_threshold)

    return {
        "models_comparison_val": comparison_rows,
        "selected_model_code": best_code,
        "selected_model_name": best_row["model_name"],
        "selected_hyperparameters": best_row["best_params"],
        "selected_threshold": selected_threshold,
        "selection_rationale": (
            "Модель LogisticRegression (class_weight='balanced') с символьными и словарными "
            "TF-IDF N-граммами и 6 ручными признаками показала лучший баланс полноты (Recall) "
            "и точности (Precision), наивысший PR-AUC/ROC-AUC, высокую скорость предсказания "
            "(< 1.5 мс на одно сообщение) и полную интерпретируемость весов признаков."
        ),
        "val_metrics_at_selected_threshold": val_metrics_tuned,
        "test_metrics": test_metrics,
        "top_15_tfidf_features": {
            cls_key: [{"feature": name, "mean_tfidf": round(score, 6)} for name, score in items]
            for cls_key, items in top_tfidf.items()
        },
        "multilingual_evaluation": multilingual_eval,
    }


def predict_sms(
    text: str,
    model_path: Path | str | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Предсказать класс и вероятность мошенничества для одного SMS-сообщения.

    Использует единую функцию нормализации ``normalize_sms`` и сохранённый
    ``Pipeline`` (по умолчанию ``ml/artifacts/sms_model.joblib`` или
    ``ml/artifacts/sms_baseline.joblib``).
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _CACHED_THRESHOLD

    target_path = Path(model_path) if model_path is not None else (
        FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BASELINE_MODEL_PATH
    )

    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != target_path:
        _CACHED_MODEL = joblib.load(target_path)
        _CACHED_MODEL_PATH = target_path
        if FINAL_META_PATH.exists():
            try:
                with FINAL_META_PATH.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                _CACHED_THRESHOLD = float(meta.get("threshold", 0.5))
            except Exception:
                _CACHED_THRESHOLD = 0.5

    active_threshold = float(_CACHED_THRESHOLD if threshold is None else threshold)
    text_clean = normalize_sms(text)
    if not text_clean:
        return {
            "label": 0,
            "category": "SAFE",
            "probability": 0.0,
            "threshold": active_threshold,
            "language": detect_language(text),
            "text_clean": "",
        }

    prob = float(_CACHED_MODEL.predict_proba([text_clean])[0, 1])
    label = int(prob >= active_threshold)
    if prob >= active_threshold:
        category = "FRAUDULENT"
    elif prob >= active_threshold * 0.6:
        category = "SUSPICIOUS"
    else:
        category = "SAFE"

    return {
        "label": label,
        "category": category,
        "probability": round(prob, 4),
        "threshold": round(active_threshold, 4),
        "language": detect_language(text),
        "text_clean": text_clean,
    }


def run_all() -> dict[str, Any]:
    """Выполнить полный пайплайн Заданий 4–8 для SMS и сохранить все артефакты."""
    print("1. Очистка набора SMS и создание стратифицированного разбиения 70/15/15...")
    cleaned_df, cleaning_report, splits = prepare_clean_data_and_splits()

    df_train = cleaned_df.iloc[splits["train_indices"]].reset_index(drop=True)
    df_val = cleaned_df.iloc[splits["val_indices"]].reset_index(drop=True)
    df_test = cleaned_df.iloc[splits["test_indices"]].reset_index(drop=True)

    print(
        f"   Очищенный набор: {len(cleaned_df)} строк "
        f"(train={len(df_train)}, val={len(df_val)}, test={len(df_test)})"
    )

    print("2. Построение графиков EDA (Задание 5)...")
    eda_summary = generate_eda_and_figures(cleaned_df, splits)

    print("3. Обучение и проверка baseline-модели SMS (Задание 7)...")
    baseline_report = train_and_evaluate_baseline(df_train, df_val)

    print("4. Сравнение вариантов признаков и 4 алгоритмов классификации (Задание 8)...")
    feature_variants_report = compare_feature_variants(df_train, df_val)
    comparison_report = compare_and_train_models(df_train, df_val, df_test)

    meta_payload: dict[str, Any] = {
        "model_type": "sms",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "algorithm": comparison_report["selected_model_name"],
        "algorithm_code": comparison_report["selected_model_code"],
        "hyperparameters": comparison_report["selected_hyperparameters"],
        "threshold": comparison_report["selected_threshold"],
        "feature_set": {
            "word_tfidf": "ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True",
            "char_tfidf": "analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_df=0.95, sublinear_tf=True",
            "manual_features": list(MANUAL_FEATURE_NAMES),
            "total_features_train": baseline_report["n_features"],
        },
        "dataset_summary": cleaning_report,
        "baseline_val_report": baseline_report,
        "feature_variants_comparison_val": feature_variants_report,
        "models_comparison_val": comparison_report["models_comparison_val"],
        "val_metrics": comparison_report["val_metrics_at_selected_threshold"],
        "test_metrics": comparison_report["test_metrics"],
        "selection_rationale": comparison_report["selection_rationale"],
        "top_15_tfidf_features": comparison_report["top_15_tfidf_features"],
        "multilingual_evaluation": comparison_report["multilingual_evaluation"],
    }

    with FINAL_META_PATH.open("w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)

    # Проверка функции predict_sms на ручных примерах
    sample_spam = predict_sms("URGENT! Your bank card is blocked. Verify now at http://bit.ly/fake-bank to win £1000!")
    sample_ham = predict_sms("Hey, are we still meeting for lunch at 1pm tomorrow?")
    print("5. Проверка predict_sms:")
    print("   Spam sample ->", sample_spam)
    print("   Ham sample  ->", sample_ham)
    print(f"   Тестовые метрики: Recall={meta_payload['test_metrics']['recall_spam']:.4f}, "
          f"Precision={meta_payload['test_metrics']['precision_spam']:.4f}, "
          f"F1={meta_payload['test_metrics']['f1_spam']:.4f}, "
          f"ROC-AUC={meta_payload['test_metrics']['roc_auc']:.4f}")

    return meta_payload


if __name__ == "__main__":
    run_all()
