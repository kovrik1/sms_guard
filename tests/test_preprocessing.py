"""Unit tests for :mod:`ml.preprocessing`."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the project root importable when pytest is invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.preprocessing import (
    clean_sms_dataframe,
    detect_language,
    extract_urls_from_text,
    mask_sensitive_text,
    normalize_sms,
)


# ---------------------------------------------------------------------------
# normalize_sms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, 123, 12.5, [], {}])
def test_normalize_non_string_returns_empty(value):
    assert normalize_sms(value) == ""


@pytest.mark.parametrize("value", ["", "   ", "\n\t"])
def test_normalize_empty_returns_empty(value):
    assert normalize_sms(value) == ""


def test_normalize_lowercases_and_applies_nfkc():
    assert normalize_sms("ПРИВЕТ") == "привет"
    assert normalize_sms("Ｈｅｌｌｏ") == "hello"


def test_normalize_replaces_email():
    out = normalize_sms("Пишите на Test.User+tag@Example.COM")
    assert out == "пишите на <email>"
    assert "test.user" not in out


@pytest.mark.parametrize(
    "raw",
    [
        "https://example.com/path?x=1",
        "www.example.com",
        "bit.ly/abc123",
        "example.kz/page",
    ],
)
def test_normalize_replaces_urls(raw):
    assert "<url>" in normalize_sms(f"Ссылка: {raw}")


@pytest.mark.parametrize(
    "raw",
    [
        "$100",
        "100$",
        "50 000 ₸",
        "500 руб",
        "100,50 р.",
        "1000 kzt",
        "1000 тенге",
        "£50",
    ],
)
def test_normalize_replaces_money(raw):
    assert "<money>" in normalize_sms(f"Цена {raw}")


@pytest.mark.parametrize(
    "raw",
    [
        "+7 (777) 123-45-67",
        "8-800-555-35-35",
        "87771234567",
        "8 (777) 123 45 67",
    ],
)
def test_normalize_replaces_phones(raw):
    assert "<phone>" in normalize_sms(f"Звони {raw}")


def test_normalize_replaces_other_numbers():
    out = normalize_sms("Код 1234, попробуйте 987654")
    assert out.count("<num>") == 2


def test_normalize_preserves_kazakh_letters():
    all_kz = "ә ғ қ ң ө ұ ү һ і Сәлем! Қазақстандағы өмір"
    out = normalize_sms(all_kz)
    for letter in ("ә", "ғ", "қ", "ң", "ө", "ұ", "ү", "һ", "і"):
        assert letter in out
    assert "сәлем" in out
    assert "қазақстандағы" in out
    assert "өмір" in out


def test_normalize_collapses_repeated_punctuation():
    assert normalize_sms("Привет!!! Как???") == "привет! как?"
    assert normalize_sms("Что?!") == "что?!"


def test_normalize_preserves_tokens_after_punctuation_cleanup():
    out = normalize_sms(
        "URL: https://a.com/x, почта a@b.com, телефон +7 (777) 123-45-67"
    )
    assert "<url>" in out
    assert "<email>" in out
    assert "<phone>" in out
    # Idempotence check: normalizing already-normalized text keeps tokens intact
    assert normalize_sms(out) == out


def test_normalize_collapses_whitespace():
    assert normalize_sms("a   b\t\tc") == "a b c"


def test_normalize_removes_stopwords():
    assert normalize_sms("Я и банк", remove_stopwords=True) == "банк"
    out = normalize_sms("это очень хороший банк", remove_stopwords=True)
    assert "это" not in out.split()
    assert "хороший" in out.split()


def test_normalize_stopwords_keep_entity_tokens():
    out = normalize_sms(
        "Ссылка https://a.com и телефон 87771234567",
        remove_stopwords=True,
    )
    assert "<url>" in out
    assert "<phone>" in out
    assert "и" not in out.split()


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

def test_detect_language_kazakh_specific_letters():
    assert detect_language("Сәлем, қалайсың?") == "kk"


def test_detect_language_kazakh_words_without_specific_letters():
    assert detect_language("аударым туралы хабарлама") == "kk"


def test_detect_language_russian():
    assert detect_language("Привет, как дела?") == "ru"


def test_detect_language_english():
    assert detect_language("Hello, how are you?") == "en"


@pytest.mark.parametrize("value", ["", "   ", "12345", "😀😀", "!!!"])
def test_detect_language_other(value):
    assert detect_language(value) == "other"


def test_detect_language_mixed_cyrillic_and_latin():
    assert detect_language("привет hello") == "ru"
    assert detect_language("hello world привет") == "en"


# ---------------------------------------------------------------------------
# mask_sensitive_text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    [
        "1234 5678 9012 3456",
        "1234-5678-9012-3456",
        "1234567890123",
    ],
)
def test_mask_bank_cards(raw):
    assert mask_sensitive_text(f"Карта {raw}") == "Карта ***"


@pytest.mark.parametrize(
    "raw",
    [
        "+7 (777) 123-45-67",
        "8-800-555-35-35",
    ],
)
def test_mask_phone_numbers(raw):
    assert mask_sensitive_text(f"Тел: {raw}") == "Тел: ***"


@pytest.mark.parametrize(
    "raw",
    [
        "50 000 ₸",
        "100,50 р.",
        "$100",
    ],
)
def test_mask_money(raw):
    assert mask_sensitive_text(f"Сумма {raw}") == "Сумма ***"


def test_mask_combined_entities():
    raw = (
        "Карта 1234 5678 9012 3456, тел +7 (777) 123-45-67, "
        "сумма 50 000 ₸"
    )
    assert mask_sensitive_text(raw) == "Карта ***, тел ***, сумма ***"


# ---------------------------------------------------------------------------
# extract_urls_from_text
# ---------------------------------------------------------------------------

def test_extract_urls_basic():
    text = "Ссылки: https://a.com/x, www.b.org, bit.ly/abc, пример.рф."
    assert extract_urls_from_text(text, max_urls=10) == [
        "https://a.com/x",
        "www.b.org",
        "bit.ly/abc",
        "пример.рф",
    ]


def test_extract_urls_respects_max_urls():
    text = "a.com b.org c.net d.kz"
    assert extract_urls_from_text(text, max_urls=2) == ["a.com", "b.org"]
    assert extract_urls_from_text(text, max_urls=0) == []


def test_extract_urls_strips_trailing_punctuation():
    text = "Открой https://example.com/test.,!?;: и всё"
    assert extract_urls_from_text(text) == ["https://example.com/test"]


def test_extract_urls_cyrillic_domain():
    assert extract_urls_from_text("Сайт пример.рф работает") == ["пример.рф"]


def test_extract_urls_returns_empty_for_non_string():
    assert extract_urls_from_text(None) == []
    assert extract_urls_from_text(123) == []


# ---------------------------------------------------------------------------
# clean_sms_dataframe
# ---------------------------------------------------------------------------

def test_clean_dataframe_basic():
    df = pd.DataFrame(
        {
            "text": [
                "Привет!",
                "Привет!",
                "",
                None,
                "Ссылка https://a.com",
                "Ещё текст",
            ],
            "label": ["ham", "ham", "spam", "spam", "spam", "ham"],
        }
    )

    cleaned, report = clean_sms_dataframe(df)

    assert len(cleaned) == 3
    assert "text_raw" in cleaned.columns
    assert "text_clean" in cleaned.columns
    assert list(cleaned["text_raw"]) == [
        "Привет!",
        "Ссылка https://a.com",
        "Ещё текст",
    ]
    assert cleaned.loc[
        cleaned["text_raw"] == "Привет!", "text_clean"
    ].iloc[0] == "привет!"
    assert cleaned.loc[
        cleaned["text_raw"] == "Ссылка https://a.com", "text_clean"
    ].iloc[0] == "ссылка <url>"

    assert report["initial_count"] == 6
    assert report["empty_removed"] == 2
    assert report["duplicates_removed"] == 1
    assert report["duplicates_by_class"] == {"ham": 1}
    assert report["final_count"] == 3


def test_clean_dataframe_with_stopwords():
    df = pd.DataFrame(
        {
            "text": ["Я и банк", "и а"],
            "label": ["ham", "spam"],
        }
    )

    cleaned, report = clean_sms_dataframe(df, remove_stopwords=True)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text_clean"] == "банк"
    assert report["empty_removed"] == 1
    assert report["final_count"] == 1


def test_clean_dataframe_custom_columns():
    df = pd.DataFrame(
        {
            "sms": ["Hello", "Hello", "World"],
            "target": [0, 0, 1],
        }
    )

    cleaned, report = clean_sms_dataframe(
        df, text_col="sms", label_col="target"
    )

    assert "text_clean" in cleaned.columns
    assert report["duplicates_by_class"] == {0: 1}
    assert report["final_count"] == 2


def test_clean_dataframe_missing_text_column_raises():
    df = pd.DataFrame({"foo": ["a"]})
    with pytest.raises(KeyError):
        clean_sms_dataframe(df)


def test_clean_dataframe_all_empty():
    df = pd.DataFrame({"text": ["", None], "label": ["a", "b"]})

    cleaned, report = clean_sms_dataframe(df)

    assert cleaned.empty
    assert report["empty_removed"] == 2
    assert report["final_count"] == 0
    assert report["duplicates_by_class"] == {}
