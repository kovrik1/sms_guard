"""Модульные тесты для :mod:`ml.predictor`.

Тесты полностью автономны: реальные артефакты ``ml/artifacts`` не
используются, все модели и метаданные создаются во временном каталоге
``tmp_path``.  Проверяются синглтон, отказоустойчивость при отсутствии
файлов (НФТ-14), формат ответов, пороги URL и отсутствие утечки текста
SMS в журналы (НФТ-9).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.predictor import (  # noqa: E402
    MODEL_NOT_LOADED_MESSAGE,
    ModelPredictor,
    predict_sms,
    predict_url,
)
from ml.url_features import URL_FEATURE_NAMES  # noqa: E402

SMS_SAFE_TEXTS = [
    "hello friend see you tomorrow at the library",
    "meeting at six near the entrance",
    "your order has been shipped and will arrive on monday",
    "happy birthday have a nice day",
    "please call me when you get home",
    "see you at the lecture hall",
    "dinner is ready come home",
    "the weather is nice today",
    "reminder about the team meeting",
    "thanks for your help yesterday",
]

SMS_SPAM_TEXTS = [
    "urgent verify your bank account now",
    "your card is blocked confirm your password",
    "claim your free prize and bonus now",
    "you have won a cash reward click the link",
    "security alert confirm your code immediately",
    "final notice transfer money to this account",
    "congratulations you are a winner claim the prize",
    "your account will be suspended update your data",
    "verify your identity and unlock your card",
    "free entry win a thousand dollars today",
]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict) -> None:
    """Записать JSON-метаданные в кодировке UTF-8."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_sms_model(path: Path) -> Pipeline:
    """Обучить небольшой текстовый pipeline (TF-IDF + LogisticRegression)."""

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    texts = SMS_SAFE_TEXTS + SMS_SPAM_TEXTS
    labels = [0] * len(SMS_SAFE_TEXTS) + [1] * len(SMS_SPAM_TEXTS)
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, path)
    return pipeline


def _build_url_model(path: Path) -> Pipeline:
    """Обучить URL-модель (StandardScaler + LogisticRegression) на 22 признаках."""

    rng = np.random.default_rng(42)
    benign = rng.normal(0.20, 0.10, size=(30, len(URL_FEATURE_NAMES)))
    malicious = rng.normal(1.20, 0.15, size=(30, len(URL_FEATURE_NAMES)))
    features = pd.DataFrame(
        np.vstack([benign, malicious]),
        columns=list(URL_FEATURE_NAMES),
    )
    labels = np.array([0] * len(benign) + [1] * len(malicious))

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(features, labels)
    joblib.dump(pipeline, path)
    return pipeline


def _fitted_dummy(constant: int) -> DummyClassifier:
    """Вернуть обученный DummyClassifier с постоянным ответом."""

    dummy = DummyClassifier(strategy="constant", constant=constant)
    dummy.fit(
        np.zeros((4, len(URL_FEATURE_NAMES))),
        np.array([0, 0, 1, 1]),
    )
    return dummy


def _build_artifacts(
    directory: Path,
    sms_version: str = "2.3.0",
    url_version: str = "3.1.4",
    sms_threshold: float = 0.5,
    t1: float = 0.40,
    t2: float = 0.70,
) -> None:
    """Создать во временном каталоге полный комплект артефактов."""

    _build_sms_model(directory / "sms_model.joblib")
    _build_url_model(directory / "url_model.joblib")
    _write_json(
        directory / "sms_model_meta.json",
        {
            "version": sms_version,
            "threshold": sms_threshold,
            "algorithm": "LogisticRegression",
        },
    )
    _write_json(
        directory / "url_model_meta.json",
        {
            "version": url_version,
            "t1": t1,
            "t2": t2,
            "algorithm": "RandomForestClassifier",
        },
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Изолировать тесты друг от друга, сбрасывая глобальный синглтон."""

    ModelPredictor.reset_instance()
    yield
    ModelPredictor.reset_instance()


# ---------------------------------------------------------------------------
# 1. Синглтон
# ---------------------------------------------------------------------------


def test_get_instance_returns_same_object_and_reset_clears_it():
    first = ModelPredictor.get_instance()
    second = ModelPredictor.get_instance()

    assert first is second

    ModelPredictor.reset_instance()
    third = ModelPredictor.get_instance()

    assert third is not first


# ---------------------------------------------------------------------------
# 2. Отсутствие артефактов (НФТ-14)
# ---------------------------------------------------------------------------


def test_empty_artifacts_dir_does_not_raise(tmp_path):
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.is_loaded is False
    assert predictor.sms_loaded is False
    assert predictor.url_loaded is False
    assert predictor.sms_model is None
    assert predictor.url_model is None
    assert predictor.get_status()["is_loaded"] is False


# ---------------------------------------------------------------------------
# 3. RuntimeError при обращении к незагруженным моделям
# ---------------------------------------------------------------------------


def test_predictions_raise_when_models_are_missing(tmp_path):
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    with pytest.raises(RuntimeError, match=MODEL_NOT_LOADED_MESSAGE):
        predictor.predict_sms("hello")
    with pytest.raises(RuntimeError, match=MODEL_NOT_LOADED_MESSAGE):
        predictor.predict_url("https://example.com")


# ---------------------------------------------------------------------------
# 4. Загрузка фиктивных обученных моделей
# ---------------------------------------------------------------------------


def test_loads_trained_models_and_exposes_them(tmp_path):
    _build_artifacts(tmp_path)

    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.sms_loaded is True
    assert predictor.url_loaded is True
    assert predictor.is_loaded is True
    assert predictor.sms_model is not None
    assert predictor.url_model is not None

    sms_result = predictor.predict_sms("urgent verify your bank account")
    url_result = predictor.predict_url("http://192.0.2.15/account-update")

    assert 0.0 <= sms_result["probability"] <= 1.0
    assert url_result["category"] in {"SAFE", "SUSPICIOUS", "DANGEROUS"}


# ---------------------------------------------------------------------------
# 5. Формат ответа predict_sms
# ---------------------------------------------------------------------------


def test_predict_sms_response_format_and_types(tmp_path):
    _build_artifacts(tmp_path, sms_threshold=0.42, sms_version="9.8.7")
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    result = predictor.predict_sms("URGENT: verify your bank account now")

    assert set(result) == {
        "probability",
        "p_sms",
        "label",
        "category",
        "threshold",
        "model_version",
    }
    assert isinstance(result["probability"], float)
    assert isinstance(result["p_sms"], float)
    assert isinstance(result["label"], int)
    assert isinstance(result["category"], str)
    assert isinstance(result["threshold"], float)
    assert isinstance(result["model_version"], str)

    assert result["probability"] == result["p_sms"]
    assert 0.0 <= result["p_sms"] <= 1.0
    assert result["threshold"] == pytest.approx(0.42)
    assert result["model_version"] == "9.8.7"
    assert result["label"] == int(result["p_sms"] >= result["threshold"])
    assert result["category"] == (
        "FRAUDULENT" if result["label"] else "SAFE"
    )


def test_predict_sms_uses_normalize_sms_by_default(tmp_path, monkeypatch):
    _build_artifacts(tmp_path)
    calls: list[str] = []

    def fake_normalize(text: str) -> str:
        calls.append(text)
        return text.lower()

    monkeypatch.setattr("ml.predictor.normalize_sms", fake_normalize)
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    predictor.predict_sms("HELLO Friend")

    assert calls == ["HELLO Friend"]


def test_predict_sms_uses_raw_text_for_text_raw_input_column(tmp_path, monkeypatch):
    _build_sms_model(tmp_path / "sms_model.joblib")
    _write_json(
        tmp_path / "sms_model_meta.json",
        {"version": "raw-1", "input_column": "text_raw", "threshold": 0.5},
    )
    calls: list[str] = []

    def fake_normalize(text: str) -> str:
        calls.append(text)
        return text.lower()

    monkeypatch.setattr("ml.predictor.normalize_sms", fake_normalize)
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    predictor.predict_sms("KEEP My CASE")

    assert calls == []


# ---------------------------------------------------------------------------
# 6. Fallback на sms_baseline.joblib
# ---------------------------------------------------------------------------


def test_falls_back_to_baseline_sms_model(tmp_path):
    _build_sms_model(tmp_path / "sms_baseline.joblib")
    _write_json(
        tmp_path / "sms_model_meta.json",
        {"version": "baseline-1.0", "threshold": 0.5},
    )

    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.sms_loaded is True
    assert predictor.url_loaded is False
    assert predictor.is_loaded is False
    assert predictor.get_status()["paths"]["sms_model_active"].endswith(
        "sms_baseline.joblib"
    )

    result = predictor.predict_sms("hello friend")
    assert result["model_version"] == "baseline-1.0"


# ---------------------------------------------------------------------------
# 7. Формат ответа predict_url
# ---------------------------------------------------------------------------


def test_predict_url_response_format_and_features(tmp_path):
    _build_artifacts(tmp_path, url_version="7.6.5")
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    result = predictor.predict_url("http://xn--exmplecua.test/secure/login?verify=1")

    assert set(result) == {
        "probability",
        "p_url",
        "category",
        "thresholds",
        "model_version",
        "features",
        "reasons",
    }
    assert isinstance(result["probability"], float)
    assert isinstance(result["p_url"], float)
    assert result["probability"] == result["p_url"]
    assert 0.0 <= result["p_url"] <= 1.0
    assert result["category"] in {"SAFE", "SUSPICIOUS", "DANGEROUS"}
    assert result["model_version"] == "7.6.5"

    assert isinstance(result["thresholds"], dict)
    assert set(result["thresholds"]) == {"t1", "t2"}
    assert result["thresholds"]["t1"] == pytest.approx(0.40)
    assert result["thresholds"]["t2"] == pytest.approx(0.70)

    assert isinstance(result["features"], dict)
    assert list(result["features"]) == list(URL_FEATURE_NAMES)
    assert len(result["features"]) == 22
    assert all(isinstance(value, (int, float)) for value in result["features"].values())

    assert isinstance(result["reasons"], list)
    assert all(isinstance(reason, str) for reason in result["reasons"])


# ---------------------------------------------------------------------------
# 8. Пороги t1 / t2 для категорий URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t1", "t2", "expected"),
    [
        (1.10, 1.20, "SAFE"),
        (0.50, 1.20, "SUSPICIOUS"),
        (0.30, 0.50, "DANGEROUS"),
    ],
)
def test_predict_url_threshold_categories(tmp_path, t1, t2, expected):
    # DummyClassifier(constant=1) всегда возвращает P_url = 1.0, поэтому
    # категория однозначно определяется только значениями порогов.
    joblib.dump(_fitted_dummy(1), tmp_path / "url_model.joblib")
    _write_json(
        tmp_path / "url_model_meta.json",
        {"version": "threshold-test", "t1": t1, "t2": t2},
    )
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    result = predictor.predict_url("https://example.com/about")

    assert result["p_url"] == pytest.approx(1.0)
    assert result["thresholds"] == {"t1": pytest.approx(t1), "t2": pytest.approx(t2)}
    assert result["category"] == expected


def test_predict_url_supports_nested_thresholds(tmp_path):
    joblib.dump(_fitted_dummy(0), tmp_path / "url_model.joblib")
    _write_json(
        tmp_path / "url_model_meta.json",
        {"version": "nested", "thresholds": {"t1": 0.25, "t2": 0.75}},
    )
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    result = predictor.predict_url("https://example.com/about")

    assert result["p_url"] == pytest.approx(0.0)
    assert result["thresholds"] == {"t1": pytest.approx(0.25), "t2": pytest.approx(0.75)}
    assert result["category"] == "SAFE"


# ---------------------------------------------------------------------------
# 9. get_model_versions() и get_status()
# ---------------------------------------------------------------------------


def test_get_model_versions_and_status(tmp_path):
    _build_artifacts(
        tmp_path,
        sms_version="2.0.0",
        url_version="3.0.0",
        sms_threshold=0.55,
        t1=0.35,
        t2=0.75,
    )
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    versions = predictor.get_model_versions()

    assert versions == {"sms": "2.0.0", "url": "3.0.0"}
    assert predictor.model_versions == versions
    assert versions is not predictor.model_versions  # возвращается копия

    status = predictor.get_status()

    assert status["is_loaded"] is True
    assert status["sms_loaded"] is True
    assert status["url_loaded"] is True
    assert status["model_versions"] == {"sms": "2.0.0", "url": "3.0.0"}
    assert status["sms_threshold"] == pytest.approx(0.55)
    assert status["url_thresholds"] == {
        "t1": pytest.approx(0.35),
        "t2": pytest.approx(0.75),
    }
    assert set(status["paths"]) == {
        "artifacts_dir",
        "sms_model",
        "sms_model_active",
        "url_model",
        "sms_meta",
        "url_meta",
    }
    assert status["paths"]["sms_model"] == str(tmp_path / "sms_model.joblib")
    assert status["paths"]["url_model"] == str(tmp_path / "url_model.joblib")


def test_model_versions_default_to_1_0_0_without_metadata(tmp_path):
    _build_sms_model(tmp_path / "sms_model.joblib")
    _build_url_model(tmp_path / "url_model.joblib")

    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.get_model_versions() == {"sms": "1.0.0", "url": "1.0.0"}


# ---------------------------------------------------------------------------
# 10. reload() без пересоздания объекта
# ---------------------------------------------------------------------------


def test_reload_picks_up_models_without_recreating_object(tmp_path):
    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.is_loaded is False

    _build_artifacts(tmp_path)

    assert predictor.reload() is True
    assert predictor.is_loaded is True
    assert predictor.sms_loaded is True
    assert predictor.url_loaded is True

    # После reload тот же экземпляр продолжает работать.
    assert predictor.predict_sms("hello friend")["model_version"] == "2.3.0"


# ---------------------------------------------------------------------------
# 11. Отсутствие текста SMS в журналах (НФТ-9)
# ---------------------------------------------------------------------------


def test_sms_text_is_not_written_to_logs(tmp_path):
    _build_artifacts(tmp_path)
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler()
    loggers = [logging.getLogger("sms_guard"), logging.getLogger("ml.predictor")]
    previous_levels = []
    for logger in loggers:
        previous_levels.append(logger.level)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    marker = "SECRET_SMS_MARKER_918273645"
    try:
        predictor = ModelPredictor(artifacts_dir=tmp_path)
        predictor.predict_sms(f"Срочно переведите 5000 {marker}")
    finally:
        for logger, level in zip(loggers, previous_levels):
            logger.removeHandler(handler)
            logger.setLevel(level)

    messages = "\n".join(record.getMessage() for record in records)

    assert messages  # загрузка моделей действительно логируется
    assert marker not in messages
    assert "Срочно переведите" not in messages


# ---------------------------------------------------------------------------
# 12. Модель без predict_proba: decision_function + сигмоида
# ---------------------------------------------------------------------------


def test_predict_sms_works_with_linear_svc(tmp_path):
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LinearSVC(max_iter=5000, random_state=42)),
        ]
    )
    texts = SMS_SAFE_TEXTS + SMS_SPAM_TEXTS
    labels = [0] * len(SMS_SAFE_TEXTS) + [1] * len(SMS_SPAM_TEXTS)
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, tmp_path / "sms_model.joblib")
    _write_json(
        tmp_path / "sms_model_meta.json",
        {"version": "svc-1.0", "threshold": 0.5},
    )

    predictor = ModelPredictor(artifacts_dir=tmp_path)
    loaded = predictor.sms_model

    assert not hasattr(loaded, "predict_proba")

    result = predictor.predict_sms("urgent verify your bank account now")

    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"] == "svc-1.0"
    assert result["label"] == int(result["probability"] >= result["threshold"])


# ---------------------------------------------------------------------------
# 13. Повреждённые артефакты не роняют инициализацию
# ---------------------------------------------------------------------------


def test_corrupted_model_file_does_not_raise(tmp_path):
    (tmp_path / "sms_model.joblib").write_bytes(b"this is not a joblib file")
    (tmp_path / "sms_model_meta.json").write_text("{broken json", encoding="utf-8")

    predictor = ModelPredictor(artifacts_dir=tmp_path)

    assert predictor.sms_loaded is False
    assert predictor.url_loaded is False
    assert predictor.is_loaded is False
    # Повреждённые метаданные не мешают значениям по умолчанию.
    assert predictor.get_model_versions() == {"sms": "1.0.0", "url": "1.0.0"}
    assert predictor.sms_threshold == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 14. Функции уровня модуля делегируют в синглтон
# ---------------------------------------------------------------------------


def test_module_level_helpers_delegate_to_singleton(tmp_path, monkeypatch):
    _build_artifacts(tmp_path)
    predictor = ModelPredictor(artifacts_dir=tmp_path)
    monkeypatch.setattr(ModelPredictor, "_instance", predictor)

    sms_result = predict_sms("hello friend")
    url_result = predict_url("https://example.com/about")

    assert sms_result["model_version"] == "2.3.0"
    assert url_result["model_version"] == "3.1.4"
