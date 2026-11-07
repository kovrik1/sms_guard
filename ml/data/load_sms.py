"""Загрузка набора данных SMS Spam Collection (UCI).

Модуль читает исходный файл ``data/raw/SMSSpamCollection`` (разделитель —
табуляция, кодировка UTF-8) и приводит его к единой структуре:

    id    — целочисленный идентификатор 1..N;
    text  — исходный текст сообщения;
    label — 0 для ham (безопасное) и 1 для spam (мошенническое).

Чтение выполняется построчно через ``str.split("\\t", 1)``, поэтому кавычки
внутри SMS не склеивают строки и не теряются. Это важно: при обычном
``pandas.read_csv`` с включённым quoting часть сообщений с кавычками
разбирается неверно, и в результате получается 5 572 строки вместо 5 574.

Запуск как скрипта::

    python -m ml.data.load_sms

сохраняет результат в ``data/processed/sms_raw.csv`` и печатает сводку.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

# Корень проекта /home/kovrik/sms_guard (ml/data/load_sms.py -> parents[2]).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_RAW_PATH: Path = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "processed" / "sms_raw.csv"

#: Соответствие текстовых меток исходного корпуса числовым меткам.
LABEL_MAPPING: dict[str, int] = {"ham": 0, "spam": 1}

EXPECTED_TOTAL_ROWS: int = 5_574
EXPECTED_HAM_ROWS: int = 4_827
EXPECTED_SPAM_ROWS: int = 747


class SmsDatasetError(ValueError):
    """Ошибка формата или кодировки исходного набора SMS."""


def _resolve_path(path: Path | str | None, default: Path) -> Path:
    """Вернуть ``Path`` для переданного значения или значение по умолчанию."""

    if path is None:
        return default
    return Path(path).expanduser()


def _read_raw_rows(raw_path: Path) -> list[tuple[str, str]]:
    """Прочитать исходный файл построчно.

    Файл открывается с ``encoding="utf-8", errors="strict"``: любая
    некорректная последовательность байтов приводит к понятной ошибке, а не
    к «тихой» замене символов. Строка разбивается ровно на две части по
    первому табулятору, поэтому табуляции внутри текста (если появятся)
    сохранятся.
    """

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Файл набора SMS не найден: {raw_path}. "
            "Ожидается data/raw/SMSSpamCollection."
        )

    rows: list[tuple[str, str]] = []
    try:
        with raw_path.open("r", encoding="utf-8", errors="strict", newline="") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                # Удаляем только перевод строки, но не пробелы в самом SMS.
                line = raw_line.rstrip("\r\n")
                if not line:
                    # Пустые строки в исходнике отсутствуют; пропускаем их,
                    # чтобы не создавать битые записи.
                    continue

                label_part, separator, text = line.partition("\t")
                if not separator:
                    raise SmsDatasetError(
                        f"Строка {line_number}: не найден разделитель '\\t' "
                        f"между меткой и текстом: {line[:80]!r}"
                    )

                label = label_part.strip().lower()
                if label not in LABEL_MAPPING:
                    raise SmsDatasetError(
                        f"Строка {line_number}: неизвестная метка {label_part!r}. "
                        f"Допустимые метки: {sorted(LABEL_MAPPING)}."
                    )

                rows.append((label, text))
    except UnicodeDecodeError as exc:
        raise SmsDatasetError(
            f"Файл {raw_path} не является корректным UTF-8: {exc}. "
            "Проверьте кодировку исходного набора."
        ) from exc

    if not rows:
        raise SmsDatasetError(f"Файл {raw_path} не содержит ни одной записи.")
    return rows


def load_sms_dataset(
    raw_path: Path | str | None = None,
    output_csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Загрузить SMS Spam Collection в ``DataFrame`` со столбцами id/text/label.

    Parameters
    ----------
    raw_path:
        Путь к исходному файлу ``SMSSpamCollection``. По умолчанию
        ``<корень проекта>/data/raw/SMSSpamCollection``.
    output_csv_path:
        Необязательный путь для сохранения результата в CSV (UTF-8,
        ``index=False``). Если не задан, файл не сохраняется.

    Returns
    -------
    pandas.DataFrame
        Столбцы ``id`` (int64, 1..N), ``text`` (str) и ``label`` (int64:
        0 — ham, 1 — spam). Число строк совпадает с числом строк исходного
        файла (5 574 для оригинального корпуса).
    """

    source = _resolve_path(raw_path, DEFAULT_RAW_PATH)
    rows = _read_raw_rows(source)

    df = pd.DataFrame(
        {
            "id": list(range(1, len(rows) + 1)),
            "text": [text for _, text in rows],
            "label": [LABEL_MAPPING[label] for label, _ in rows],
        }
    )
    df = df.astype({"id": "int64", "text": "object", "label": "int64"})

    if output_csv_path is not None:
        save_sms_dataset(df, output_csv_path)

    return df


def save_sms_dataset(
    df: pd.DataFrame,
    output_csv_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Сохранить ``DataFrame`` в CSV UTF-8 без индекса и вернуть путь."""

    destination = _resolve_path(output_csv_path, DEFAULT_OUTPUT_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, encoding="utf-8")
    return destination


def summarize_sms_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Напечатать и вернуть сводку по набору SMS.

    Возвращаемый словарь содержит ключи ``total_rows``, ``dtypes``,
    ``class_distribution`` (абсолютные числа и проценты), ``examples_ham``
    и ``examples_spam`` (по 5 примеров каждого класса).
    """

    required_columns = {"id", "text", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise SmsDatasetError(
            f"В DataFrame отсутствуют обязательные столбцы: {sorted(missing)}."
        )

    total_rows = int(len(df))
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}

    class_distribution: dict[str, dict[str, float | int | str]] = {}
    for name, label in (("ham", LABEL_MAPPING["ham"]), ("spam", LABEL_MAPPING["spam"])):
        count = int((df["label"] == label).sum())
        percent = round(100.0 * count / total_rows, 2) if total_rows else 0.0
        class_distribution[name] = {
            "label": label,
            "count": count,
            "percent": percent,
        }

    examples_ham = df.loc[df["label"] == LABEL_MAPPING["ham"], "text"].head(5).tolist()
    examples_spam = df.loc[df["label"] == LABEL_MAPPING["spam"], "text"].head(5).tolist()

    summary: dict[str, Any] = {
        "total_rows": total_rows,
        "dtypes": dtypes,
        "class_distribution": class_distribution,
        "examples_ham": examples_ham,
        "examples_spam": examples_spam,
    }

    # --- Вывод в консоль ---------------------------------------------------
    print(f"Всего строк: {total_rows}")
    print("Типы столбцов:")
    print(df.dtypes.to_string())

    print("Распределение классов:")
    for name, stats in class_distribution.items():
        print(f"  {name}: {stats['count']} ({stats['percent']:.2f}%)")

    print("Примеры ham (безопасные):")
    for index, text in enumerate(examples_ham, start=1):
        print(f"  {index}. {text}")
    print("Примеры spam (мошеннические/спам):")
    for index, text in enumerate(examples_spam, start=1):
        print(f"  {index}. {text}")

    return summary


def _read_with_pandas(raw_path: Path) -> pd.DataFrame:
    """Альтернативное чтение через ``pandas.read_csv`` без quoting.

    Оставлено как явная демонстрация корректного варианта из ТЗ:
    ``sep="\\t"``, ``header=None``, ``quoting=csv.QUOTE_NONE``. Основной
    загрузчик использует построчное чтение, но эта функция полезна для
    сверки числа строк.
    """

    data = pd.read_csv(
        raw_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        engine="python",
    )
    data["label"] = data["label"].str.strip().str.lower().map(LABEL_MAPPING)
    if data["label"].isna().any():
        bad = data.loc[data["label"].isna(), "label"].unique().tolist()
        raise SmsDatasetError(f"Неизвестные метки в файле: {bad}")
    data["label"] = data["label"].astype("int64")
    data.insert(0, "id", range(1, len(data) + 1))
    return data[["id", "text", "label"]].astype(
        {"id": "int64", "text": "object", "label": "int64"}
    )


#: Алиас для совместимости с ноутбуками и другими заданиями.
load_sms = load_sms_dataset


def main() -> None:
    """Точка входа скрипта: загрузка, сохранение и сводка."""

    df = load_sms_dataset(output_csv_path=DEFAULT_OUTPUT_PATH)
    print(f"Данные сохранены в: {DEFAULT_OUTPUT_PATH}")
    summarize_sms_dataset(df)


if __name__ == "__main__":
    main()
