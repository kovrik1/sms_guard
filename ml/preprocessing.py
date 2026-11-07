"""Text preprocessing utilities for the SMS Guard project.

The module exposes a small set of pure helpers used by the ML pipeline:

* :func:`normalize_sms` - normalize a single SMS message.
* :func:`detect_language` - detect ``ru`` / ``kk`` / ``en`` / ``other``.
* :func:`mask_sensitive_text` - mask cards, phones and money amounts.
* :func:`extract_urls_from_text` - extract a limited list of URLs.
* :func:`clean_sms_dataframe` - clean a pandas DataFrame and build a report.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

# All Cyrillic letters used in Russian and Kazakh (lower case).
_CYR = "а-яёәғқңөұүһі"

_LABEL = rf"[a-z0-9{_CYR}](?:[a-z0-9{_CYR}\-]*[a-z0-9{_CYR}])?"
_DOMAIN = rf"(?:{_LABEL}\.)+"

_TLDS = (
    r"(?:kz|ru|com|org|net|xyz|top|click|example|test|"
    r"рф|рус|онлайн|сайт|москва|дети|ком|нет|орг)"
)
_SHORTENERS = r"(?:bit\.ly|t\.co|clck\.ru|vk\.cc|tinyurl\.com|cutt\.ly|is\.gd|goo\.gl)"

_EMAIL_RE = re.compile(
    r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
    re.IGNORECASE,
)

_URL_RE = re.compile(
    rf"(?:(?:https?://|www\.)[^\s<>\"']+"
    rf"|{_SHORTENERS}/[^\s<>\"']*"
    rf"|{_DOMAIN}{_TLDS}(?::\d+)?(?:/[^\s<>\"']*)?"
    rf"|{_DOMAIN}[a-z{_CYR}]{{2,}}/[^\s<>\"']*)",
    re.IGNORECASE | re.UNICODE,
)

_CURRENCY = (
    r"(?:\$|£|€|₸|"
    r"(?<![а-яёa-z])(?:руб(?:лей|ля|ль|\.)?|тенге|р\.|тг|tenge|usd|kzt)"
    r"(?![а-яёa-z]))"
)
_MONEY_NUM = r"\d+(?:(?:[ \u00a0\u202f]\d{3})|(?:[.,]\d{3}))*(?:[.,]\d{1,2})?"
_MONEY_RE = re.compile(
    rf"(?:{_CURRENCY}[ \u00a0\u202f]*{_MONEY_NUM}"
    rf"|{_MONEY_NUM}[ \u00a0\u202f]*{_CURRENCY})",
    re.IGNORECASE | re.UNICODE,
)

_PHONE_CAND_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[ \u00a0\u202f\-]?)?"
    r"(?:\(\d{2,4}\)[ \u00a0\u202f\-]?)?"
    r"\d[\d \u00a0\u202f\-\(\)\.]{7,20}\d(?!\d)",
    re.UNICODE,
)

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

_NON_KEEP_RE = re.compile(r"[^\w\s!?]", re.UNICODE)

_CARD_CAND_RE = re.compile(
    r"(?<!\d)(?:\d[ \u00a0\u202f\-]?){12,18}\d(?!\d)",
    re.UNICODE,
)

_URL_EXTRACT_RE = re.compile(
    rf"(?:(?<![@\w.\-])(?:https?://|www\.)[^\s<>\"']+"
    rf"|(?<![@\w.\-]){_SHORTENERS}/[^\s<>\"']*"
    rf"|(?<![@\w.\-]){_DOMAIN}{_TLDS}(?::\d+)?(?:/[^\s<>\"']*)?"
    rf"|(?<![@\w.\-]){_DOMAIN}[a-z{_CYR}]{{2,}}/[^\s<>\"']*)",
    re.IGNORECASE | re.UNICODE,
)

_TOKEN_MAP = (
    ("__TOKEN_EMAIL__", "<email>"),
    ("__TOKEN_URL__", "<url>"),
    ("__TOKEN_MONEY__", "<money>"),
    ("__TOKEN_PHONE__", "<phone>"),
    ("__TOKEN_NUM__", "<num>"),
)

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

_STOPWORDS: set[str] = {
    # Russian
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том",
    "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между",
    "это", "очень", "также", "такие", "такое", "кроме", "сразу", "однако",
    "поэтому", "ваш", "ваши", "вашего", "вашей", "вашу", "свой", "свои",
    "своих", "своего", "своей", "наш", "наши", "наших", "нашего", "нашей",
    "нашу", "который", "которая", "которые", "которых", "которого", "которой",
    "которую", "которым", "которыми",
    # English
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "as", "from", "that", "this", "these", "those", "it",
    "its", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "their", "our", "my", "me", "him", "us", "do", "does", "did", "have",
    "has", "had", "not", "no", "yes", "so", "just", "can", "will", "would",
    "could", "should", "may", "might", "must", "here", "there", "what",
    "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "only",
    "own", "same", "than", "too", "very", "s", "t", "don", "now",
    # Kazakh (basic)
    "және", "бірақ", "немесе", "үшін", "бұл", "осы", "сол", "бар", "жоқ",
    "емес", "де", "да", "ме", "ба", "па", "пе", "ма", "сонымен", "яғни",
    "өйткені", "қалай", "не", "кім", "қайда", "қашан", "неге", "өте", "көп",
    "аз", "барлық", "әр", "бір", "екі", "үш", "мен", "сен", "ол", "біз",
    "сіз", "олар", "менің", "сенің", "оның", "біздің", "сіздің", "олардың",
}

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_KAZAKH_SPECIFIC = set("әғқңөұүһі")
_KAZAKH_WORDS = {
    "сәлем",
    "теңге",
    "картаңыз",
    "сілтеме",
    "жедел",
    "растаңыз",
    "бұғатталды",
    "құпия",
    "шотыңыз",
    "банкке",
    "аударым",
    "кодты",
}
_WORD_RE = re.compile(rf"[a-z{_CYR}]+", re.UNICODE)


# ---------------------------------------------------------------------------
# Replacement callbacks
# ---------------------------------------------------------------------------

def _phone_placeholder(match: re.Match[str]) -> str:
    """Replace a candidate with ``<phone>`` only if it looks like a phone."""
    candidate = match.group(0)
    digits = re.sub(r"\D", "", candidate)
    if 10 <= len(digits) <= 15:
        return " __TOKEN_PHONE__ "
    return candidate


def _mask_card(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = re.sub(r"\D", "", candidate)
    if 13 <= len(digits) <= 19:
        return "***"
    return candidate


def _mask_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = re.sub(r"\D", "", candidate)
    if 10 <= len(digits) <= 15:
        return "***"
    return candidate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_sms(text: str, remove_stopwords: bool = False) -> str:
    """Normalize a single SMS message.

    The function is pure: it never mutates its input and always returns a
    string.  Non-string values and blank strings produce an empty string.

    Args:
        text: Raw SMS text.
        remove_stopwords: If ``True``, drop Russian, English and basic Kazakh
            stop words.  Entity tokens and punctuation are always preserved.

    Returns:
        A normalized lower-case string with entity tokens such as
        ``<url>``, ``<email>``, ``<money>``, ``<phone>`` and ``<num>``.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = unicodedata.normalize("NFKC", text).lower()

    # Preserve already-existing entity tokens so normalize_sms is idempotent.
    for placeholder, token in _TOKEN_MAP:
        text = text.replace(token, f" {placeholder} ")

    # Replace entities by digit-free placeholders first.  The placeholders are
    # surrounded by spaces so that adjacent words do not get glued together.
    text = _EMAIL_RE.sub(" __TOKEN_EMAIL__ ", text)
    text = _URL_RE.sub(" __TOKEN_URL__ ", text)
    text = _MONEY_RE.sub(" __TOKEN_MONEY__ ", text)
    text = _PHONE_CAND_RE.sub(_phone_placeholder, text)
    text = _NUM_RE.sub(" __TOKEN_NUM__ ", text)

    # Remove control characters, emoji and punctuation, but keep letters
    # (including all Kazakh letters), digits, whitespace, ``!`` and ``?``.
    text = _NON_KEEP_RE.sub(" ", text)

    # Collapse repeated punctuation and whitespace.
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([!?])", r"\1", text)
    text = re.sub(r"([!?])(?=[^\s!?])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        text = " ".join(
            word for word in text.split() if word.lower() not in _STOPWORDS
        )

    # Convert temporary placeholders into the public angle-bracket tokens.
    for placeholder, token in _TOKEN_MAP:
        text = text.replace(placeholder, token)

    # Drop stray underscores that were not part of a placeholder.
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Detect the language of ``text``.

    Returns one of ``"ru"``, ``"kk"``, ``"en"`` or ``"other"``.
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    lower = text.lower()

    # Kazakh-specific letters are the strongest signal.
    if any(ch in _KAZAKH_SPECIFIC for ch in lower):
        return "kk"

    # Some Kazakh words do not contain Kazakh-specific letters.
    words = set(_WORD_RE.findall(lower))
    if words & _KAZAKH_WORDS:
        return "kk"

    cyrillic = sum(1 for ch in lower if "а" <= ch <= "я" or ch == "ё")
    latin = sum(1 for ch in lower if "a" <= ch <= "z")

    if cyrillic > 0 and cyrillic >= latin:
        return "ru"
    if latin > 0:
        return "en"
    return "other"


def mask_sensitive_text(text: str) -> str:
    """Mask bank cards, phone numbers and money amounts with ``***``."""
    if not isinstance(text, str) or not text:
        return ""

    masked = _CARD_CAND_RE.sub(_mask_card, text)
    masked = _PHONE_CAND_RE.sub(_mask_phone, masked)
    masked = _MONEY_RE.sub("***", masked)
    return masked


def extract_urls_from_text(text: str, max_urls: int = 3) -> list[str]:
    """Extract up to ``max_urls`` URLs from ``text``.

    Trailing punctuation such as ``.``, ``,``, ``!``, ``?``, ``;`` and ``:`` is
    removed.  Duplicates are dropped while preserving the original order.
    """
    if not isinstance(text, str) or not text:
        return []
    if not isinstance(max_urls, int) or max_urls <= 0:
        return []

    urls: list[str] = []
    seen: set[str] = set()

    for match in _URL_EXTRACT_RE.finditer(text):
        url = match.group(0).rstrip(".,!?;:)'\"")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
            if len(urls) >= max_urls:
                break

    return urls


def clean_sms_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    remove_stopwords: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean an SMS DataFrame and return ``(cleaned_df, report)``.

    The returned DataFrame contains ``text_raw`` (the original text) and
    ``text_clean`` (the normalized text).  Rows with empty text (before or
    after normalization) and duplicate cleaned messages are removed.
    """
    if text_col not in df.columns:
        raise KeyError(f"Column {text_col!r} not found in DataFrame")

    cleaned = df.copy()
    initial_count = len(cleaned)

    cleaned["text_raw"] = cleaned[text_col]
    cleaned["text_clean"] = cleaned[text_col].apply(
        lambda value: normalize_sms(value, remove_stopwords=remove_stopwords)
    )

    def _is_empty(value: Any) -> bool:
        return not isinstance(value, str) or not value.strip()

    empty_mask = cleaned[text_col].apply(_is_empty) | (cleaned["text_clean"] == "")
    empty_removed = int(empty_mask.sum())
    cleaned = cleaned.loc[~empty_mask].copy()

    duplicate_mask = cleaned.duplicated(subset=["text_clean"], keep="first")
    duplicates_removed = int(duplicate_mask.sum())

    duplicates_by_class: dict[Any, int] = {}
    if label_col in cleaned.columns:
        value_counts = cleaned.loc[duplicate_mask, label_col].value_counts(
            dropna=False
        )
        for label, count in value_counts.items():
            key: Any = label
            if pd.isna(key):
                key = "nan"
            elif hasattr(key, "item"):
                key = key.item()
            duplicates_by_class[key] = int(count)

    cleaned = cleaned.loc[~duplicate_mask].copy().reset_index(drop=True)

    report = {
        "initial_count": initial_count,
        "empty_removed": empty_removed,
        "duplicates_removed": duplicates_removed,
        "duplicates_by_class": duplicates_by_class,
        "final_count": len(cleaned),
    }
    return cleaned, report
