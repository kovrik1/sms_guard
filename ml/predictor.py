"""Загрузка обученных ML-моделей и предсказание для SMS и URL.

Модуль реализует потокобезопасный синглтон :class:`ModelPredictor`
(Задание 12, НФТ-1, НФТ-9, НФТ-14) и две удобные функции уровня модуля:
:func:`predict_sms` и :func:`predict_url`.

Ключевые свойства реализации:

* модели загружаются с диска один раз на процесс — при создании синглтона
  или первом вызове :meth:`ModelPredictor.get_instance`;
* отсутствие или повреждение файлов моделей не приводит к падению импорта
  или инициализации: свойства ``is_loaded`` / ``sms_loaded`` / ``url_loaded``
  просто возвращают ``False`` (НФТ-14);
* тексты SMS никогда не попадают в журналы: логируются только пути к
  артефактам, версии моделей и факт успешной/неуспешной загрузки (НФТ-9);
* все пути по умолчанию строятся относительно корня проекта, поэтому модуль
  не зависит от текущей рабочей директории.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.preprocessing import normalize_sms
from ml.url_features import (
    URL_FEATURE_NAMES,
    explain_url,
    extract_url_features,
)

# Корень проекта: ml/predictor.py -> parents[1].
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Каталог артефактов по умолчанию.
DEFAULT_ARTIFACTS_DIR: Path = PROJECT_ROOT / "ml" / "artifacts"

#: Имя сообщения об ошибке при обращении к незагруженной модели.
MODEL_NOT_LOADED_MESSAGE = "Model artifacts are not loaded"

#: Логгеры, в которые пишутся сообщения о загрузке моделей.
_LOGGER_NAMES: tuple[str, str] = ("sms_guard", "ml.predictor")

_DEFAULT_SMS_THRESHOLD = 0.5
_DEFAULT_URL_T1 = 0.40
_DEFAULT_URL_T2 = 0.70
_DEFAULT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _log(level: int, message: str, *args: Any) -> None:
    """Записать сообщение сразу в оба логгера проекта.

    Сообщения содержат только служебные сведения (пути, версии, коды
    ошибок) и никогда — пользовательский текст SMS (НФТ-9).
    """

    for name in _LOGGER_NAMES:
        logging.getLogger(name).log(level, message, *args)


def _django_settings() -> Any:
    """Вернуть объект настроек Django, если он сконфигурирован, иначе ``None``."""

    try:
        from django.conf import settings  # noqa: PLC0415 (мягкая зависимость)
    except Exception:  # pragma: no cover — Django не установлен
        return None

    try:
        if settings.configured:
            return settings
    except Exception:  # pragma: no cover — защита от нестандартной конфигурации
        return None
    return None


def _get_setting(name: str, default: Any = None) -> Any:
    """Прочитать атрибут ``django.conf.settings``, если Django настроен."""

    settings = _django_settings()
    if settings is None:
        return default
    return getattr(settings, name, default)


def _as_path(value: Any) -> Path | None:
    """Преобразовать значение в ``Path``; пустые значения дают ``None``."""

    if value is None or value == "":
        return None
    try:
        return Path(value).expanduser()
    except TypeError:
        return None


def _coerce_float(value: Any, default: float) -> float:
    """Привести значение к ``float``, вернув ``default`` при ошибке."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _sigmoid(score: float) -> float:
    """Устойчивая к переполнению логистическая функция."""

    if score >= 0.0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def _clip_probability(value: float) -> float:
    """Ограничить вероятность диапазоном ``[0.0, 1.0]``."""

    return float(min(1.0, max(0.0, value)))


def _positive_probability(model: Any, X: Any, fallback_X: Any = None) -> float:
    """Вычислить вероятность положительного класса (класс 1).

   Сначала используется ``predict_proba``, затем — ``decision_function``
    с сигмоидой.  Для моделей, не поддерживающих ни один из методов,
    выбрасывается :class:`RuntimeError`.
    """

    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            proba = predict_proba(X)
        except Exception:
            if fallback_X is None:
                raise
            proba = predict_proba(fallback_X)
        array = np.asarray(proba, dtype=float)
        if array.ndim == 2 and array.shape[1] >= 2:
            return _clip_probability(float(array[0, 1]))
        return _clip_probability(float(array.ravel()[0]))

    decision_function = getattr(model, "decision_function", None)
    if callable(decision_function):
        try:
            scores = decision_function(X)
        except Exception:
            if fallback_X is None:
                raise
            scores = decision_function(fallback_X)
        array = np.asarray(scores, dtype=float).ravel()
        if array.size == 0:
            raise RuntimeError("Model returned an empty decision_function result")
        return _clip_probability(_sigmoid(float(array[0])))

    raise RuntimeError("Model does not support predict_proba or decision_function")


def _is_usable_model(model: Any) -> bool:
    """Проверить, что загруженный объект похож на классификатор."""

    if model is None:
        return False
    for name in ("predict_proba", "decision_function"):
        if callable(getattr(model, name, None)):
            return True
    return False


def _read_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-словарь метаданных; при ошибке вернуть пустой словарь."""

    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        _log(
            logging.WARNING,
            "Не удалось прочитать метаданные модели %s: %s",
            path,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        _log(
            logging.WARNING,
            "Файл метаданных модели %s не содержит JSON-объект",
            path,
        )
        return {}
    return data


# ---------------------------------------------------------------------------
# Синглтон ModelPredictor
# ---------------------------------------------------------------------------


class ModelPredictor:
    """Потокобезопасный синглтон для загрузки ML-моделей и предсказаний.

    Экземпляр хранит обученные модели в памяти и не перечитывает их с диска
    на каждый запрос (НФТ-1).  Загрузка артефактов выполняется в
    :meth:`__init__`; :meth:`get_instance` создаёт единственный экземпляр на
    процесс под защитой :class:`threading.RLock`.

    Порядок разрешения путей:

    1. явные аргументы ``artifacts_dir`` / ``sms_model_path`` /
       ``url_model_path``;
    2. атрибуты ``django.conf.settings`` (``ML_ARTIFACTS_DIR``,
       ``SMS_MODEL_PATH``, ``URL_MODEL_PATH``), если Django настроен;
    3. значения по умолчанию относительно :data:`DEFAULT_ARTIFACTS_DIR`.
    """

    _instance: "ModelPredictor | None" = None
    _instance_lock: threading.RLock = threading.RLock()

    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        sms_model_path: str | Path | None = None,
        url_model_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._sms_model: Any = None
        self._url_model: Any = None
        self._sms_meta: dict[str, Any] = {}
        self._url_meta: dict[str, Any] = {}

        self.artifacts_dir = self._resolve_artifacts_dir(artifacts_dir)
        self.sms_model_path = self._resolve_model_path(
            sms_model_path, "SMS_MODEL_PATH", "sms_model.joblib"
        )
        self.url_model_path = self._resolve_model_path(
            url_model_path, "URL_MODEL_PATH", "url_model.joblib"
        )
        self.sms_meta_path = self.artifacts_dir / "sms_model_meta.json"
        self.url_meta_path = self.artifacts_dir / "url_model_meta.json"
        self._active_sms_model_path = self.sms_model_path

        with self._lock:
            self._load_all()

    # -- разрешение путей ---------------------------------------------------

    def _resolve_artifacts_dir(self, value: str | Path | None) -> Path:
        """Определить каталог артефактов с учётом настроек Django."""

        explicit = _as_path(value)
        if explicit is not None:
            return explicit
        from_settings = _as_path(_get_setting("ML_ARTIFACTS_DIR"))
        if from_settings is not None:
            return from_settings
        return DEFAULT_ARTIFACTS_DIR

    def _resolve_model_path(
        self,
        value: str | Path | None,
        setting_name: str,
        filename: str,
    ) -> Path:
        """Определить путь к файлу модели с учётом настроек Django."""

        explicit = _as_path(value)
        if explicit is not None:
            return explicit
        from_settings = _as_path(_get_setting(setting_name))
        if from_settings is not None:
            return from_settings
        return self.artifacts_dir / filename

    # -- загрузка моделей ---------------------------------------------------

    def _load_all(self) -> bool:
        """Загрузить обе модели и метаданные; вернуть ``self.is_loaded``."""

        self._sms_meta = self._load_meta(self.sms_meta_path)
        self._url_meta = self._load_meta(self.url_meta_path)

        self._sms_model = self._load_sms_model()
        self._url_model = self._load_model(self.url_model_path, "URL")

        versions = self.get_model_versions()
        if self.is_loaded:
            _log(
                logging.INFO,
                "ML-модели загружены: sms=%s (v%s), url=%s (v%s)",
                self._active_sms_model_path,
                versions["sms"],
                self.url_model_path,
                versions["url"],
            )
        else:
            _log(
                logging.WARNING,
                "ML-модели загружены не полностью: sms_loaded=%s, url_loaded=%s",
                self.sms_loaded,
                self.url_loaded,
            )
        return self.is_loaded

    def _load_sms_model(self) -> Any:
        """Загрузить SMS-модель, при необходимости — baseline-артефакт."""

        candidate = self.sms_model_path
        if not candidate.exists() and candidate.name == "sms_model.joblib":
            baseline = candidate.parent / "sms_baseline.joblib"
            if baseline.exists():
                _log(
                    logging.INFO,
                    "Основная SMS-модель не найдена, используется fallback: %s",
                    baseline,
                )
                candidate = baseline

        self._active_sms_model_path = candidate
        return self._load_model(candidate, "SMS")

    def _load_model(self, path: Path, kind: str) -> Any:
        """Загрузить один joblib-артефакт, не позволяя исключению всплыть."""

        if not path.exists():
            _log(
                logging.WARNING,
                "Файл %s-модели не найден: %s",
                kind,
                path,
            )
            return None

        try:
            import joblib  # noqa: PLC0415 (импорт внутри функции ускоряет старт)
        except Exception as exc:  # pragma: no cover — joblib есть в зависимостях
            _log(
                logging.ERROR,
                "Библиотека joblib недоступна, %s-модель не загружена: %s",
                kind,
                exc,
            )
            return None

        try:
            model = joblib.load(path)
        except Exception as exc:
            _log(
                logging.ERROR,
                "Не удалось загрузить %s-модель из %s: %s",
                kind,
                path,
                exc,
            )
            return None

        if not _is_usable_model(model):
            _log(
                logging.ERROR,
                "Файл %s-модели %s не содержит классификатор",
                kind,
                path,
            )
            return None

        return model

    @staticmethod
    def _load_meta(path: Path) -> dict[str, Any]:
        """Загрузить метаданные модели, вернув пустой словарь при ошибке."""

        if not path.exists():
            _log(logging.WARNING, "Файл метаданных не найден: %s", path)
            return {}
        return _read_json(path)

    # -- управление синглтоном ---------------------------------------------

    @classmethod
    def get_instance(
        cls,
        artifacts_dir: str | Path | None = None,
    ) -> "ModelPredictor":
        """Вернуть единственный экземпляр класса, создав его при первом вызове."""

        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(artifacts_dir=artifacts_dir)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Сбросить сохранённый экземпляр синглтона (используется в тестах)."""

        with cls._instance_lock:
            cls._instance = None

    def reload(self) -> bool:
        """Перечитать модели и метаданные с диска, вернуть ``self.is_loaded``."""

        with self._lock:
            _log(logging.INFO, "Перезагрузка ML-моделей из %s", self.artifacts_dir)
            return self._load_all()

    # -- состояние ----------------------------------------------------------

    @property
    def sms_model(self) -> Any:
        """Загруженная SMS-модель или ``None``."""

        return self._sms_model

    @property
    def url_model(self) -> Any:
        """Загруженная URL-модель или ``None``."""

        return self._url_model

    @property
    def sms_meta(self) -> dict[str, Any]:
        """Метаданные SMS-модели."""

        return dict(self._sms_meta)

    @property
    def url_meta(self) -> dict[str, Any]:
        """Метаданные URL-модели."""

        return dict(self._url_meta)

    @property
    def sms_loaded(self) -> bool:
        """Признак успешной загрузки SMS-модели."""

        return self._sms_model is not None

    @property
    def url_loaded(self) -> bool:
        """Признак успешной загрузки URL-модели."""

        return self._url_model is not None

    @property
    def is_loaded(self) -> bool:
        """``True``, только если загружены обе модели."""

        return self.sms_loaded and self.url_loaded

    @property
    def model_versions(self) -> dict[str, str]:
        """Версии моделей в виде ``{'sms': ..., 'url': ...}``."""

        return {
            "sms": self._extract_version(self._sms_meta),
            "url": self._extract_version(self._url_meta),
        }

    def get_model_versions(self) -> dict[str, str]:
        """Вернуть копию словаря версий моделей."""

        return self.model_versions

    @staticmethod
    def _extract_version(meta: dict[str, Any]) -> str:
        """Извлечь версию из метаданных, используя значение по умолчанию."""

        for key in ("version", "model_version"):
            value = meta.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return _DEFAULT_VERSION

    @property
    def sms_threshold(self) -> float:
        """Порог классификации SMS (по умолчанию 0.5)."""

        return _coerce_float(
            self._sms_meta.get("threshold"), _DEFAULT_SMS_THRESHOLD
        )

    @property
    def url_thresholds(self) -> dict[str, float]:
        """Пороги URL-модели ``{'t1': ..., 't2': ...}``."""

        nested = self._url_meta.get("thresholds")
        nested = nested if isinstance(nested, dict) else {}
        t1 = self._url_meta.get("t1", nested.get("t1", _DEFAULT_URL_T1))
        t2 = self._url_meta.get("t2", nested.get("t2", _DEFAULT_URL_T2))
        return {
            "t1": _coerce_float(t1, _DEFAULT_URL_T1),
            "t2": _coerce_float(t2, _DEFAULT_URL_T2),
        }

    def get_status(self) -> dict[str, Any]:
        """Вернуть словарь с текущим статусом загрузки моделей."""

        return {
            "is_loaded": self.is_loaded,
            "sms_loaded": self.sms_loaded,
            "url_loaded": self.url_loaded,
            "model_versions": self.get_model_versions(),
            "sms_threshold": self.sms_threshold,
            "url_thresholds": self.url_thresholds,
            "paths": {
                "artifacts_dir": str(self.artifacts_dir),
                "sms_model": str(self.sms_model_path),
                "sms_model_active": str(self._active_sms_model_path),
                "url_model": str(self.url_model_path),
                "sms_meta": str(self.sms_meta_path),
                "url_meta": str(self.url_meta_path),
            },
        }

    # -- предсказания -------------------------------------------------------

    def predict_sms(self, text: str) -> dict[str, Any]:
        """Классифицировать SMS-сообщение.

        Args:
            text: Исходный текст сообщения.

        Returns:
            Словарь с вероятностью ``probability`` / ``p_sms``, меткой
            ``label``, категорией ``category``, порогом ``threshold`` и
            версией модели ``model_version``.

        Raises:
            RuntimeError: если SMS-модель не загружена.
        """

        model = self._sms_model
        if model is None:
            raise RuntimeError(MODEL_NOT_LOADED_MESSAGE)

        raw_text = "" if text is None else str(text)
        input_column = str(self._sms_meta.get("input_column", "")).strip()
        if input_column == "text_raw":
            model_input = raw_text
        else:
            model_input = normalize_sms(raw_text)
            # For non-English SMS without social engineering indicators,
            # remove isolated <num> placeholders before English TF-IDF so that
            # L2-normalization on a single OOV-surviving token ('num') does not
            # artificially inflate p_sms on benign times/dates (e.g. '18:00', '4821').
            try:
                from ml.preprocessing import detect_language
                if detect_language(raw_text) != "en":
                    has_se = False
                    try:
                        from ml.rules import detect_se_rules
                        has_se = bool(detect_se_rules(raw_text))
                    except Exception:
                        has_se = False
                    # Крупные суммы ("50 000", "30 000", 5+ цифр) — признак
                    # мошенничества: плейсхолдеры <num> в таких SMS не вырезаем.
                    has_amount = bool(
                        re.search(r"\d{1,3}(?:[\s\u00a0.,]\d{3})+|\d{5,}", raw_text)
                    )
                    if (
                        not has_se
                        and not has_amount
                        and "<url>" not in model_input
                        and "<money>" not in model_input
                    ):
                        model_input = re.sub(r"<num>|\bnum\b", "", model_input).strip()
            except Exception:
                pass

        probability = _positive_probability(model, [model_input])
        threshold = self.sms_threshold
        label = int(probability >= threshold)

        return {
            "probability": float(probability),
            "p_sms": float(probability),
            "label": label,
            "category": "FRAUDULENT" if label else "SAFE",
            "threshold": float(threshold),
            "model_version": self._extract_version(self._sms_meta),
        }

    def predict_url(self, url: str) -> dict[str, Any]:
        """Классифицировать URL по 22 структурным признакам.

        Args:
            url: Строка URL (сеть не используется).

        Returns:
            Словарь с вероятностью ``probability`` / ``p_url``, категорией,
            порогами, версией модели, признаками и объяснениями.

        Raises:
            RuntimeError: если URL-модель не загружена.
        """

        model = self._url_model
        if model is None:
            raise RuntimeError(MODEL_NOT_LOADED_MESSAGE)

        features = extract_url_features(url)
        reasons = explain_url(features)

        frame = pd.DataFrame(
            [[float(features[name]) for name in URL_FEATURE_NAMES]],
            columns=list(URL_FEATURE_NAMES),
        )
        numpy_input = frame.to_numpy(dtype=float)

        probability = _positive_probability(model, frame, fallback_X=numpy_input)
        thresholds = self.url_thresholds
        t1 = thresholds["t1"]
        t2 = thresholds["t2"]

        if probability < t1:
            category = "SAFE"
        elif probability < t2:
            category = "SUSPICIOUS"
        else:
            category = "DANGEROUS"

        return {
            "probability": float(probability),
            "p_url": float(probability),
            "category": category,
            "thresholds": {"t1": float(t1), "t2": float(t2)},
            "model_version": self._extract_version(self._url_meta),
            "features": features,
            "reasons": reasons,
        }


# ---------------------------------------------------------------------------
# Удобные функции уровня модуля (Задание 7 п. 6)
# ---------------------------------------------------------------------------


def predict_sms(text: str) -> dict[str, Any]:
    """Делегировать предсказание SMS синглтону :class:`ModelPredictor`."""

    return ModelPredictor.get_instance().predict_sms(text)


def predict_url(url: str) -> dict[str, Any]:
    """Делегировать предсказание URL синглтону :class:`ModelPredictor`."""

    return ModelPredictor.get_instance().predict_url(url)


__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "MODEL_NOT_LOADED_MESSAGE",
    "PROJECT_ROOT",
    "ModelPredictor",
    "predict_sms",
    "predict_url",
]
