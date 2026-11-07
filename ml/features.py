"""Извлечение признаков из SMS (задание 6).

Модуль содержит:

* ``create_sms_splits`` — стратифицированное разбиение 70/15/15 с
  сохранением индексов в ``data/processed/sms_split.json``;
* ``ManualSmsFeatures`` — сериализуемый sklearn-трансформер с шестью
  ручными признаками;
* ``build_sms_feature_union`` — объединение словной и символьной TF-IDF
  с ручными признаками;
* ``get_top_tfidf_features`` — топ-N признаков по среднему TF-IDF для
  классов ham/spam.

Все пути строятся относительно корня проекта, поэтому модуль не зависит
от текущей рабочей директории. Векторизаторы и масштабирование обучаются
только внутри ``Pipeline``/``FeatureUnion`` на обучающей выборке: вызовы
``fit`` на val/test не выполняются.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

# Корень проекта /home/kovrik/sms_guard (ml/features.py -> parents[1]).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH: Path = PROJECT_ROOT / "data" / "processed" / "sms_split.json"

#: Имена шести ручных признаков в порядке столбцов матрицы.
MANUAL_FEATURE_NAMES: tuple[str, ...] = (
    "char_length",
    "caps_ratio",
    "exclamation_count",
    "has_url",
    "has_money_or_phone",
    "se_groups_count",
)

#: Признаки наличия URL: служебный токен и типовые подстроки.
_URL_MARKERS: tuple[str, ...] = ("<url>", "http://", "https://", "www.", "bit.ly")

#: Признаки наличия денег/телефона: служебные токены, валюты и подстроки.
_MONEY_MARKERS: tuple[str, ...] = (
    "<money>",
    "<phone>",
    "$",
    "£",
    "€",
    "₸",
    "руб",
    "тг",
)

_PHONE_PATTERN = re.compile(r"\d{7,}")


def _compile_keyword(keyword: str) -> re.Pattern[str]:
    """Скомпилировать шаблон ключевого слова с учётом границ слова."""

    keyword = keyword.strip()
    if not keyword:
        raise ValueError("Пустое ключевое слово в словаре социальной инженерии.")
    if re.fullmatch(r"[\w \-']+", keyword, flags=re.UNICODE):
        # Ограничиваем слово границами, чтобы "now" не срабатывало в "know",
        # а "win" — в "window".
        return re.compile(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", re.IGNORECASE)
    return re.compile(re.escape(keyword), re.IGNORECASE)


#: Встроенный fallback-словарь из шести групп социальной инженерии.
#: Используется, если ``ml.rules`` ещё не создан или ``detect_se_rules``
#: недоступен. Ключевые слова покрывают английский, русский и казахский.
_FALLBACK_SE_GROUPS: dict[str, tuple[str, ...]] = {
    "urgency": (
        "urgent",
        "immediately",
        "hurry",
        "asap",
        "expire",
        "last chance",
        "final notice",
        "act now",
        "right now",
        "срочно",
        "немедленно",
        "шұғыл",
    ),
    "money": (
        "money",
        "cash",
        "transfer",
        "payment",
        "payout",
        "refund",
        "$",
        "£",
        "€",
        "₸",
        "руб",
        "тг",
        "тенге",
        "деньги",
        "перевод",
        "оплата",
    ),
    "bank": (
        "bank",
        "banking",
        "credit card",
        "debit card",
        "visa",
        "mastercard",
        "iban",
        "карта",
        "банк",
        "счет",
        "шот",
    ),
    "account_block": (
        "blocked",
        "block your",
        "suspend",
        "suspended",
        "locked",
        "deactivat",
        "unblock",
        "заблок",
        "блокир",
        "аккаунт",
        "есептік жазба",
    ),
    "prize": (
        "win",
        "winner",
        "prize",
        "reward",
        "bonus",
        "claim",
        "award",
        "congratulation",
        "free entry",
        "выигр",
        "приз",
        "бонус",
        "ұтыс",
    ),
    "code_request": (
        "otp",
        "pin code",
        "password",
        "verification code",
        "confirmation code",
        "security code",
        "one-time code",
        "код",
        "пароль",
        "пин-код",
        "растау коды",
    ),
}

_FALLBACK_SE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    group: tuple(_compile_keyword(keyword) for keyword in keywords)
    for group, keywords in _FALLBACK_SE_GROUPS.items()
}


def _resolve_path(path: Path | str | None, default: Path) -> Path:
    """Вернуть ``Path`` для переданного значения или значение по умолчанию."""

    if path is None:
        return default
    return Path(path).expanduser()


def _count_matched_groups(result: Any) -> int:
    """Привести результат ``detect_se_rules`` к числу сработавших групп."""

    if result is None:
        return 0
    # bool проверяем до int, потому что bool является подклассом int.
    if isinstance(result, bool | np.bool_):
        return int(result)
    if isinstance(result, Mapping):
        for key in ("matched_groups", "groups", "matches"):
            if key in result:
                return _count_matched_groups(result[key])
        return sum(1 for value in result.values() if value)
    if isinstance(result, set | frozenset):
        return len(result)
    if isinstance(result, list | tuple):
        matched = 0
        for item in result:
            if isinstance(item, Mapping):
                for key in ("matched", "is_match", "is_matched", "hit"):
                    if key in item:
                        matched += int(bool(item[key]))
                        break
                else:
                    # Словарь без явного флага считается сработавшей группой.
                    matched += 1
            else:
                for attribute in ("matched", "is_match", "is_matched", "hit"):
                    if hasattr(item, attribute):
                        matched += int(bool(getattr(item, attribute)))
                        break
                else:
                    matched += int(bool(item))
        return matched
    if isinstance(result, int | np.integer):
        return int(result)
    try:
        return len(result)
    except TypeError:
        return int(bool(result))


def _count_fallback_se_groups(text: str) -> int:
    """Посчитать число групп словаря, сработавших в тексте."""

    matched = 0
    for patterns in _FALLBACK_SE_PATTERNS.values():
        if any(pattern.search(text) for pattern in patterns):
            matched += 1
    return matched


def _count_se_groups(text: str) -> int:
    """Посчитать группы социальной инженерии через ``ml.rules`` или fallback."""

    try:
        from ml.rules import detect_se_rules  # noqa: PLC0415 (ленивый импорт)
    except Exception:  # модуль ещё не создан или сломан импорт
        return _count_fallback_se_groups(text)

    if not callable(detect_se_rules):
        return _count_fallback_se_groups(text)

    try:
        result = detect_se_rules(text)
    except Exception:  # несовместимая сигнатура/ошибка правил
        return _count_fallback_se_groups(text)
    return _count_matched_groups(result)


class ManualSmsFeatures(BaseEstimator, TransformerMixin):
    """Ручные признаки SMS, совместимые с ``joblib`` и ``Pipeline``.

    Для каждого текста вычисляются шесть числовых признаков:

    =================== ===============================================
    Признак             Смысл
    =================== ===============================================
    ``char_length``     длина текста в символах;
    ``caps_ratio``      доля заглавных букв среди букв текста;
    ``exclamation_count`` число символов ``!`` и ``?``;
    ``has_url``         есть ли ``<url>``/http(s)/www./bit.ly;
    ``has_money_or_phone`` есть ли ``<money>``/``<phone>``/валюта/7+ цифр;
    ``se_groups_count`` число сработавших групп соц. инженерии.
    =================== ===============================================

    Класс не содержит лямбда-функций и иных несериализуемых атрибутов,
    поэтому сохраняется и загружается через ``joblib`` без ограничений.
    """

    def fit(self, X: Iterable[str], y: Any = None) -> ManualSmsFeatures:
        """Ничего не вычисляет: ручные признаки не требуют обучения."""

        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        """Преобразовать итерируемый объект текстов в матрицу (n, 6)."""

        texts = list(X)
        matrix = np.empty((len(texts), len(MANUAL_FEATURE_NAMES)), dtype=np.float64)

        for row, raw_text in enumerate(texts):
            text = "" if raw_text is None else str(raw_text)
            lowered = text.lower()

            letters = [char for char in text if char.isalpha()]
            caps_ratio = (
                sum(1 for char in letters if char.isupper()) / len(letters)
                if letters
                else 0.0
            )

            has_url = 1.0 if any(marker in lowered for marker in _URL_MARKERS) else 0.0

            has_money_or_phone = (
                1.0
                if any(marker in lowered for marker in _MONEY_MARKERS)
                or _PHONE_PATTERN.search(text) is not None
                else 0.0
            )

            matrix[row] = (
                float(len(text)),
                float(caps_ratio),
                float(text.count("!") + text.count("?")),
                has_url,
                has_money_or_phone,
                float(_count_se_groups(text)),
            )

        return matrix

    def get_feature_names_out(
        self, input_features: Iterable[str] | None = None
    ) -> np.ndarray:
        """Вернуть имена шести ручных признаков."""

        return np.asarray(MANUAL_FEATURE_NAMES, dtype=object)


def create_sms_splits(
    df: pd.DataFrame,
    output_path: Path | str | None = None,
    random_state: int = 42,
) -> dict[str, list[int]]:
    """Стратифицированно разбить SMS на train/val/test в пропорции 70/15/15.

    Parameters
    ----------
    df:
        ``DataFrame`` с обязательным целочисленным столбцом ``label``
        (0 — ham, 1 — spam).
    output_path:
        Путь для JSON с индексами. По умолчанию
        ``data/processed/sms_split.json``.
    random_state:
        Зерно генератора случайных чисел (по умолчанию 42).

    Returns
    -------
    dict
        ``{"train_indices": [...], "val_indices": [...],
        "test_indices": [...]}``. Для целочисленного ``df.index``
        сохраняются его значения, иначе используются позиционные индексы.
        Все элементы приводятся к ``int``.
    """

    if "label" not in df.columns:
        raise KeyError("В DataFrame отсутствует обязательный столбец 'label'.")
    if len(df) < 4:
        raise ValueError("Для стратифицированного разбиения нужно минимум 4 строки.")

    labels = df["label"].to_numpy()

    # Разбиение выполняем по позициям: так стратификация всегда выровнена
    # с целевым массивом, даже если df.index не является RangeIndex.
    positions = np.arange(len(df))

    # 70% train, 30% temp (val + test) со стратификацией.
    train_positions, temp_positions = train_test_split(
        positions,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )

    # 30% делим пополам: 15% val и 15% test.
    val_positions, test_positions = train_test_split(
        temp_positions,
        test_size=0.50,
        random_state=random_state,
        stratify=labels[temp_positions],
    )

    # Для целочисленного df.index возвращаем его значения, иначе позиции.
    if pd.api.types.is_integer_dtype(df.index.dtype):
        index_values = df.index.to_numpy()
    else:
        index_values = positions

    splits: dict[str, list[int]] = {
        "train_indices": sorted(int(index) for index in index_values[train_positions]),
        "val_indices": sorted(int(index) for index in index_values[val_positions]),
        "test_indices": sorted(int(index) for index in index_values[test_positions]),
    }

    destination = _resolve_path(output_path, DEFAULT_SPLIT_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(splits, fh, ensure_ascii=False, indent=2)

    return splits


split_sms_dataset = create_sms_splits


def build_sms_feature_union(
    remove_stopwords: bool = False,
    include_char_ngrams: bool = True,
) -> FeatureUnion:
    """Собрать ``FeatureUnion`` из TF-IDF и ручных признаков.

    Состав:

    1. ``word_tfidf`` — словарные униграммы и биграммы;
    2. ``char_tfidf`` — символьные N-граммы ``char_wb`` (3–5), если
       ``include_char_ngrams=True``;
    3. ``manual_features`` — ``ManualSmsFeatures`` + ``MinMaxScaler``.

    Возвращаемый объект не обучен: ``fit`` должен вызываться только на
    обучающей выборке.
    """

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english" if remove_stopwords else None,
    )

    transformers: list[tuple[str, Any]] = [("word_tfidf", word_tfidf)]

    if include_char_ngrams:
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        transformers.append(("char_tfidf", char_tfidf))

    manual_pipeline = Pipeline(
        [
            ("extract", ManualSmsFeatures()),
            ("scale", MinMaxScaler()),
        ]
    )
    transformers.append(("manual_features", manual_pipeline))

    return FeatureUnion(transformers)


def _get_feature_names(fitted_union: FeatureUnion) -> np.ndarray:
    """Получить имена признаков обученного ``FeatureUnion``."""

    try:
        names = fitted_union.get_feature_names_out()
    except Exception as exc:  # pragma: no cover — защита от старых sklearn
        raise RuntimeError(
            "Не удалось получить имена признаков FeatureUnion. "
            "Убедитесь, что все трансформеры поддерживают "
            "get_feature_names_out()."
        ) from exc
    return np.asarray(names, dtype=object)


def get_top_tfidf_features(
    fitted_union: FeatureUnion,
    X_train_texts: Iterable[str],
    y_train: Iterable[int] | None,
    top_n: int = 15,
) -> dict[str, list[tuple[str, float]]]:
    """Топ-N признаков по среднему TF-IDF отдельно для ham и spam.

    Если ``fitted_union`` ещё не обучен, он обучается **только** на
    переданных обучающих текстах ``X_train_texts``/``y_train``. Валидация
    и тест сюда не передаются, поэтому утечки данных не происходит.

    Returns
    -------
    dict
        ``{"ham": [(name, mean_score), ...], "spam": [...]}``.
    """

    if top_n <= 0:
        raise ValueError("top_n должен быть положительным целым числом.")

    texts = list(X_train_texts)
    try:
        matrix = fitted_union.transform(texts)
    except NotFittedError:
        if y_train is None:
            raise ValueError(
                "FeatureUnion не обучен, а y_train не передан: невозможно "
                "выполнить fit только на обучающей выборке."
            ) from None
        fitted_union.fit(texts, y_train)
        matrix = fitted_union.transform(texts)

    names = _get_feature_names(fitted_union)
    labels = np.asarray([] if y_train is None else y_train)

    # Ручные признаки не являются TF-IDF; в рейтинг «топ TF-IDF» они не
    # попадают, иначе MinMax-масштабированные значения вытесняют слова.
    tfidf_mask = ~np.char.startswith(names.astype(str), "manual_features__")
    if tfidf_mask.any():
        names = names[tfidf_mask]
        matrix = matrix[:, tfidf_mask]

    result: dict[str, list[tuple[str, float]]] = {"ham": [], "spam": []}
    for class_label, key in ((0, "ham"), (1, "spam")):
        mask = labels == class_label
        if not mask.any():
            continue

        class_matrix = matrix[mask]
        if hasattr(class_matrix, "toarray"):
            # Для разреженной матрицы mean(axis=0) возвращает np.matrix.
            mean_scores = np.asarray(class_matrix.mean(axis=0)).ravel()
        else:
            mean_scores = np.asarray(class_matrix, dtype=np.float64).mean(axis=0)

        order = np.argsort(-mean_scores, kind="stable")[:top_n]
        result[key] = [
            (str(names[index]), float(mean_scores[index])) for index in order
        ]

    return result
