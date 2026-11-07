"""
Модульные тесты для извлечения признаков SMS (ml/features.py) и загрузки данных
(ml/data/load_sms.py, ml/data/load_url.py) — Задания 3, 6, 10, НФТ-15.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.load_sms import (
    SmsDatasetError,
    _read_raw_rows,
    _read_with_pandas,
    load_sms_dataset,
    save_sms_dataset,
    summarize_sms_dataset,
)
from ml.data.load_url import (
    _build_synthetic_url_samples,
    _deartifact_safe_urls,
    _extract_domain_group,
    _normalize_url_case,
)
from ml.features import (
    MANUAL_FEATURE_NAMES,
    ManualSmsFeatures,
    _compile_keyword,
    _count_fallback_se_groups,
    _count_matched_groups,
    _count_se_groups,
    build_sms_feature_union,
    create_sms_splits,
    get_top_tfidf_features,
)


# ==============================================================================
# 1. Тесты ml/features.py (Задание 6)
# ==============================================================================


def test_manual_sms_features_fit_transform_and_names():
    """Проверка ManualSmsFeatures: размерность (n, 6), значения признаков и get_feature_names_out."""
    transformer = ManualSmsFeatures()
    texts = [
        "Привет, встречаемся завтра в парке",
        "URGENT!! переведите 5000 ₸ на +77012345678 или http://bank.example/login",
        None,
        "12345",
    ]
    fitted = transformer.fit(texts)
    assert fitted is transformer

    mat = transformer.transform(texts)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (4, 6)

    # Второй текст содержит заглавные буквы, '!', URL, деньги/телефон и группы СИ
    assert mat[1, 0] > 20  # char_length
    assert mat[1, 1] > 0.0  # caps_ratio
    assert mat[1, 2] >= 2.0  # exclamation_count
    assert mat[1, 3] == 1.0  # has_url
    assert mat[1, 4] == 1.0  # has_money_or_phone
    assert mat[1, 5] >= 1.0  # se_groups_count

    # Пустой / None текст
    assert mat[2, 0] == 0.0
    assert mat[2, 1] == 0.0

    names = transformer.get_feature_names_out()
    assert tuple(names) == MANUAL_FEATURE_NAMES


def test_helper_count_matched_groups_and_fallback():
    """Проверка всех веток _count_matched_groups, _count_fallback_se_groups и _compile_keyword."""
    assert _count_matched_groups(None) == 0
    assert _count_matched_groups(True) == 1
    assert _count_matched_groups(False) == 0
    assert _count_matched_groups(3) == 3
    assert _count_matched_groups({"matched_groups": ["URGENCY", "PRIZE"]}) == 2
    assert _count_matched_groups({"URGENCY": True, "PRIZE": False, "BANK": 1}) == 2
    assert _count_matched_groups({"a", "b", "c"}) == 3
    assert (
        _count_matched_groups(
            [
                {"code": "URGENCY", "matched": True},
                {"code": "PRIZE", "matched": False},
                {"code": "BANK"},
            ]
        )
        == 2
    )

    class DummyRule:
        def __init__(self, hit: bool):
            self.hit = hit

    assert _count_matched_groups([DummyRule(True), DummyRule(False), " raw "]) == 2
    assert (
        _count_fallback_se_groups("URGENT! You won a prize! Send money to bank card code")
        >= 4
    )
    assert _count_se_groups("Срочно назовите код из SMS для разблокировки карты") >= 2

    with pytest.raises(ValueError):
        _compile_keyword("   ")


def test_create_sms_splits_stratified_and_errors(tmp_path: Path):
    """Проверка стратифицированного разбиения 70/15/15 и сохранения JSON."""
    df = pd.DataFrame(
        {
            "id": range(1, 41),
            "text": [f"msg {i}" for i in range(40)],
            "label": [0] * 30 + [1] * 10,
        }
    )
    split_file = tmp_path / "sms_split.json"
    splits = create_sms_splits(df, output_path=split_file, random_state=42)

    assert split_file.exists()
    train_idx = set(splits["train_indices"])
    val_idx = set(splits["val_indices"])
    test_idx = set(splits["test_indices"])

    assert len(train_idx) == 28
    assert len(val_idx) == 6
    assert len(test_idx) == 6
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)

    # Проверка со строковым индексом DataFrame
    df_str_idx = df.set_index(pd.Index([f"row_{i}" for i in range(40)]))
    splits2 = create_sms_splits(df_str_idx, output_path=tmp_path / "split2.json")
    assert len(splits2["train_indices"]) == 28

    # Ошибки валидации
    with pytest.raises(KeyError):
        create_sms_splits(pd.DataFrame({"text": ["a", "b", "c", "d"]}), output_path=split_file)
    with pytest.raises(ValueError):
        create_sms_splits(pd.DataFrame({"label": [0, 1]}), output_path=split_file)


def test_build_sms_feature_union_and_top_tfidf():
    """Проверка сборки FeatureUnion и расчёта топ-признаков TF-IDF по классам."""
    texts = [
        "привет как дела встречаемся завтра в библиотеке",
        "привет завтра пара по математике в университете",
        "доброе утро жду тебя у входа в парк завтра",
        "спасибо за помощь увидимся вечером дома",
        "срочно ваша карта заблокирована подтвердите код по ссылке",
        "срочно банк заблокировал счет введите код подтверждения по ссылке",
        "вы выиграли приз переведите деньги на карту срочно",
        "поздравляем вы выиграли приз заберите бонус по ссылке банк",
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    union = build_sms_feature_union(remove_stopwords=False, include_char_ngrams=True)
    # Вызов get_top_tfidf_features на ещё не обученном union должен обучить его на train
    top_features = get_top_tfidf_features(union, texts, labels, top_n=5)
    assert "ham" in top_features and "spam" in top_features
    assert len(top_features["ham"]) == 5
    assert len(top_features["spam"]) == 5

    # Повторный вызов на уже обученном union
    top_features_2 = get_top_tfidf_features(union, texts, labels, top_n=3)
    assert len(top_features_2["spam"]) == 3

    # Вариант без char_ngrams и со стоп-словами
    union_words_only = build_sms_feature_union(
        remove_stopwords=True, include_char_ngrams=False
    )
    mat = union_words_only.fit_transform(texts)
    assert mat.shape[0] == len(texts)

    with pytest.raises(ValueError):
        get_top_tfidf_features(union, texts, labels, top_n=0)

    unfitted = build_sms_feature_union()
    with pytest.raises(ValueError):
        get_top_tfidf_features(unfitted, texts, y_train=None, top_n=5)


# ==============================================================================
# 2. Тесты ml/data/load_sms.py (Задание 3)
# ==============================================================================


def test_load_save_and_summarize_sms_dataset(tmp_path: Path):
    """Проверка чтения TSV, сохранения CSV, summarize_sms_dataset и _read_with_pandas."""
    raw_tsv = tmp_path / "SMSSpamCollection"
    raw_tsv.write_text(
        "ham\tПривет, увидимся завтра в 18:00\n"
        "spam\tURGENT! Claim your prize at http://prize.example\n"
        "\n"
        "ham\tКупи хлеб по дороге домой\n"
        "spam\tВаша карта заблокирована, назовите код из SMS\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "sms_raw.csv"

    df = load_sms_dataset(raw_path=raw_tsv, output_csv_path=out_csv)
    assert out_csv.exists()
    assert list(df.columns) == ["id", "text", "label"]
    assert len(df) == 4
    assert df["label"].tolist() == [0, 1, 0, 1]

    summary = summarize_sms_dataset(df)
    assert summary["total_rows"] == 4
    assert summary["class_distribution"]["ham"]["count"] == 2
    assert summary["class_distribution"]["spam"]["count"] == 2

    df_pandas = _read_with_pandas(raw_tsv)
    assert len(df_pandas) == 4
    assert df_pandas["label"].tolist() == [0, 1, 0, 1]

    saved_path = save_sms_dataset(df, tmp_path / "sub" / "saved.csv")
    assert saved_path.exists()


def test_load_sms_dataset_error_handling(tmp_path: Path):
    """Проверка обработки ошибок формата, кодировки и отсутствия файла в load_sms.py."""
    with pytest.raises(FileNotFoundError):
        _read_raw_rows(tmp_path / "nonexistent_file")

    empty_file = tmp_path / "empty.tsv"
    empty_file.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SmsDatasetError):
        _read_raw_rows(empty_file)

    no_tab_file = tmp_path / "no_tab.tsv"
    no_tab_file.write_text("ham message without tab\n", encoding="utf-8")
    with pytest.raises(SmsDatasetError):
        _read_raw_rows(no_tab_file)

    bad_label_file = tmp_path / "bad_label.tsv"
    bad_label_file.write_text("unknown_label\tsome text\n", encoding="utf-8")
    with pytest.raises(SmsDatasetError):
        _read_raw_rows(bad_label_file)
    with pytest.raises(SmsDatasetError):
        _read_with_pandas(bad_label_file)

    bad_utf8_file = tmp_path / "bad_utf8.tsv"
    bad_utf8_file.write_bytes(b"ham\t\xff\xfeinvalid utf8\n")
    with pytest.raises(SmsDatasetError):
        _read_raw_rows(bad_utf8_file)

    with pytest.raises(SmsDatasetError):
        summarize_sms_dataset(pd.DataFrame({"text": ["hello"]}))


# ==============================================================================
# 3. Тесты вспомогательных функций ml/data/load_url.py (Задание 10)
# ==============================================================================


def test_load_url_helpers_and_deartifacting():
    """Проверка нормализации URL, извлечения доменной группы и устранения артефактов PhiUSIIL."""
    assert _normalize_url_case("") == ""
    assert _normalize_url_case("HTTPS://WWW.EXAMPLE.COM/Login?A=1") == "https://www.example.com/Login?A=1"
    assert _normalize_url_case("EXAMPLE.COM/Path") == "example.com/Path"

    assert _extract_domain_group("https://sub.kaspi.kz/pay") == "kaspi.kz"
    assert _extract_domain_group("http://192.0.2.15/login") == "192.0.2.15"

    rng = np.random.default_rng(42)
    safe_urls = [f"https://www.site-{i}.example" for i in range(40)]
    deart_urls, stats = _deartifact_safe_urls(safe_urls, rng)
    assert len(deart_urls) == 40
    assert stats["www_removed"] > 0
    assert stats["safe_path_added"] > 0

    synth = _build_synthetic_url_samples(rng)
    assert len(synth) > 1000
    labels = {label for _, label in synth}
    assert labels == {0, 1}
