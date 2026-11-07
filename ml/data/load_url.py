"""Загрузка, очистка, устранение артефактов разметки и построение признаков URL (Задание 10).

Модуль читает открытый датасет PhiUSIIL Phishing URL Dataset, приводит метки к
бинарному стандарту проекта (0 — безопасный/SAFE, 1 — фишинг/DANGEROUS),
устраняет структурные артефакты разметки PhiUSIIL с помощью непрерывного
генератора путей/параметров (длина 1–60 символов, 1–4 сегмента, слова, числовые
сегменты, расширения, query), помечает синтетические примеры флагом
``is_synthetic``, выполняет групповое разбиение 70/15/15 по регистрируемому
домену (``GroupShuffleSplit``) и извлекает 22 признака функцией
:func:`ml.url_features.extract_url_features`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.url_features import (
    BRAND_DOMAINS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
    URL_FEATURE_NAMES,
    _TLD_EXTRACTOR,
    extract_url_features,
    validate_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
DEFAULT_PARQUET_PATH = REPO_ROOT / "data" / "processed" / "url_features.parquet"
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "processed" / "url_split.json"

_SAFE_WORDS_POOL = (
    "about", "docs", "news", "catalog", "ru", "en", "kk", "person", "help",
    "search", "mail", "u", "wiki", "articles", "questions", "downloads",
    "release", "python", "cpython", "org", "project", "shop", "services",
    "info", "company", "support", "terms", "privacy", "cards", "tracking",
    "credits", "personal", "profile", "faq", "contact", "index", "home",
    "products", "item", "view", "page", "post", "blog", "guide", "reference",
    "api", "v1", "v2", "v3", "manual", "tutorial", "status", "press",
)

_PHISH_SEG_POOL = (
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "wallet", "unlock", "pay", "card", "bonus", "prize", "password", "bank",
    "suspend", "free", "kaspi", "halyk", "egov",
)

_SUBDOMAIN_POOL = (
    "docs", "mail", "ru", "en", "kk", "news", "help", "support", "wiki", "blog", "my", "app",
)

_EXT_POOL = ("", "", "", "", ".html", ".php", ".htm", ".aspx")


def _normalize_url_case(url: str) -> str:
    """Привести схему и хост URL к нижнему регистру, сохранив путь и запрос."""
    raw = str(url).strip()
    if not raw:
        return ""
    if "://" not in raw and not raw.startswith("//"):
        parsed = urlparse("http://" + raw)
        host = (parsed.netloc or "").lower()
        rest = urlunparse(("", "", parsed.path, parsed.params, parsed.query, ""))
        return f"{host}{rest}"
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "http").lower()
    netloc = (parsed.netloc or "").lower()
    return urlunparse((scheme, netloc, parsed.path, parsed.params, parsed.query, ""))


def _extract_domain_group(url: str) -> str:
    """Извлечь регистрируемый домен (или hostname) для группового разбиения."""
    is_valid, normalized, _ = validate_url(url)
    target = normalized if is_valid else ("http://" + str(url).strip())
    try:
        parsed = urlparse(target)
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        hostname = str(url).strip().lower()

    if not hostname:
        return "unknown"

    ext = _TLD_EXTRACTOR(hostname)
    if hasattr(ext, "top_domain_under_public_suffix"):
        reg = (ext.top_domain_under_public_suffix or "").lower()
    else:
        reg = (getattr(ext, "registered_domain", "") or "").lower()
    return reg if reg else hostname


def _generate_safe_path_and_query(rng: np.random.Generator) -> tuple[str, str]:
    """Сгенерировать разнообразный безопасный путь (длина непрерывно 1..60 символов,
    1..4 сегмента, слова, числовые сегменты /3/, /0/, /2024/05/, repo-пути
    org/project, расширения .html/.php) и опциональный query."""
    if rng.random() < 0.06:
        return "/", ""

    pattern = int(rng.integers(0, 7))
    if pattern == 0:
        # Короткий числовой или языковой сегмент: /3/, /0/, /ru/, /docs/
        seg = str(
            rng.choice(
                ["0", "1", "2", "3", "4", "5", "ru", "en", "kk", "u", "v1", "v2", "v3", "docs", "about"]
            )
        )
        trailing = "/" if rng.random() < 0.70 else ""
        path = f"/{seg}{trailing}"
    elif pattern == 1:
        # Многосегментный путь с короткими индексами (/mail/u/0/, /ru/person, /docs/3/)
        w1 = str(rng.choice(["mail", "docs", "ru", "en", "kk", "u", "news", "help", "catalog", "api", "wiki"]))
        w2 = str(
            rng.choice(["u", "person", "cards", "tracking", "about", "0", "1", "2", "3", "v1", "v2", "python"])
        )
        if rng.random() < 0.55:
            w3 = str(
                rng.choice(["0", "1", "2", "3", "index", "inbox", "overview", "person", "list", "cpython"])
            )
            trailing = "/" if rng.random() < 0.65 else ""
            path = f"/{w1}/{w2}/{w3}{trailing}"
        else:
            trailing = "/" if rng.random() < 0.50 else ""
            path = f"/{w1}/{w2}{trailing}"
    elif pattern == 2:
        # Репозитории и разделы сайта вида /org/project, /python/cpython, /wiki/Python (без цифр)
        n_seg = int(rng.integers(1, 4))
        segs = []
        for _ in range(n_seg):
            w = str(rng.choice(_SAFE_WORDS_POOL))
            if rng.random() < 0.22:
                w = f"{w}-{rng.choice(_SAFE_WORDS_POOL)}"
            segs.append(w)
        trailing = "/" if rng.random() < 0.40 else str(rng.choice(_EXT_POOL))
        path = "/" + "/".join(segs) + trailing
    elif pattern == 3:
        # Статьи и карточки товаров с числовым ID: /ru/articles/123456/, /catalog/item/4821
        w1 = str(rng.choice(["ru", "en", "kk", "news", "blog", "articles", "questions", "catalog", "item", "docs"]))
        num_digits = int(rng.integers(1, 9))
        low = 10 ** (num_digits - 1) if num_digits > 1 else 0
        num_str = str(int(rng.integers(low, 10**num_digits)))
        if rng.random() < 0.60:
            w2 = str(rng.choice(["articles", "posts", "item", "view", "release", "questions", "doc", "u"]))
            trailing = "/" if rng.random() < 0.60 else str(rng.choice(_EXT_POOL))
            path = f"/{w1}/{w2}/{num_str}{trailing}"
        else:
            trailing = "/" if rng.random() < 0.60 else str(rng.choice(_EXT_POOL))
            path = f"/{w1}/{num_str}{trailing}"
    elif pattern == 4:
        # Архивы по датам и релизы: /2024/05/news-item.html, /downloads/release/python-3120/
        if rng.random() < 0.50:
            year = int(rng.integers(2018, 2026))
            month = f"{int(rng.integers(1, 13)):02d}"
            slug = str(rng.choice(_SAFE_WORDS_POOL))
            if rng.random() < 0.50:
                slug += f"-{int(rng.integers(10, 9999))}"
            path = f"/{year}/{month}/{slug}" + ("/" if rng.random() < 0.50 else ".html")
        else:
            w1 = str(rng.choice(["downloads", "docs", "catalog", "archive", "products"]))
            w2 = str(rng.choice(["release", "version", "item", "package", "series"]))
            ver = int(rng.integers(100, 99999))
            path = f"/{w1}/{w2}/python-{ver}/"
    elif pattern == 5:
        # Длинные человекочитаемые пути (StackOverflow / Habr / блог, длина 30..60 символов)
        w1 = str(rng.choice(["questions", "articles", "blog", "news", "docs", "catalog"]))
        num_id = int(rng.integers(1000, 99999999))
        words = [str(rng.choice(_SAFE_WORDS_POOL)) for _ in range(int(rng.integers(3, 7)))]
        slug = "-".join(words)
        path = f"/{w1}/{num_id}/{slug}"
        if len(path) > 60:
            path = path[:60].rstrip("-")
    else:
        # Непрерывное покрытие всей сетки длин 2..60 символов
        target_len = int(rng.integers(2, 61))
        segs = []
        while len("/" + "/".join(segs)) < target_len:
            if rng.random() < 0.42:
                segs.append(str(int(rng.integers(0, 999999))))
            else:
                segs.append(str(rng.choice(_SAFE_WORDS_POOL)))
        path = ("/" + "/".join(segs))[:target_len]
        if len(path) < 2:
            path = "/a"

    query = ""
    if rng.random() < 0.22:
        qtype = int(rng.integers(0, 4))
        if qtype == 0:
            query = f"q={rng.choice(['test', 'weather', 'python', 'news', 'info'])}&page={int(rng.integers(1, 9))}"
        elif qtype == 1:
            query = f"id={int(rng.integers(1, 99999))}"
        elif qtype == 2:
            query = f"lang={rng.choice(['ru', 'kk', 'en'])}"
        else:
            query = f"q={rng.choice(['search', 'docs', 'item'])}"
    return path, query


def _generate_phish_path_and_query(rng: np.random.Generator) -> tuple[str, str]:
    """Сгенерировать путь для фишинговых URL без пути, чтобы выровнять распределение
    path_length и num_digits_url между классами без создания обратного артефакта."""
    w1 = str(rng.choice(_PHISH_SEG_POOL))
    w2 = str(rng.choice(_PHISH_SEG_POOL))
    n_seg = int(rng.integers(1, 4))
    if n_seg == 1:
        path = f"/{w1}" + ("/" if rng.random() < 0.40 else ".php")
    elif n_seg == 2:
        if rng.random() < 0.45:
            num = int(rng.integers(10, 99999))
            path = f"/{w1}/{w2}-{num}" + ("/" if rng.random() < 0.40 else ".html")
        else:
            path = f"/{w1}/{w2}" + ("/" if rng.random() < 0.50 else "")
    else:
        num = int(rng.integers(1, 999999))
        path = f"/{w1}/{w2}/{num}" + ("/" if rng.random() < 0.40 else ".php")
    if len(path) > 60:
        path = path[:60]

    query = ""
    if rng.random() < 0.25:
        query = f"{rng.choice(['verify', 'token', 'id', 'session'])}={int(rng.integers(1, 99999))}"
    return path, query


def _deartifact_safe_urls(urls: list[str], rng: np.random.Generator) -> tuple[list[str], dict[str, int]]:
    """Устранить артефакты разметки PhiUSIIL на безопасных URL.

    1. У ~50% безопасных URL убирается ``www.`` (и у ~25% из них добавляется
       типичный поддомен ``docs.``, ``mail.``, ``ru.``, ``en.``, ``wiki.``).
    2. У ~15% безопасных URL схема ``https://`` заменяется на ``http://``.
    3. К ~68% безопасных URL добавляется разнообразный путь длиной 1..60 символов
       из генератора :func:`_generate_safe_path_and_query`.
    """
    out: list[str] = []
    stats = {"www_removed": 0, "subdomain_varied": 0, "scheme_to_http": 0, "safe_path_added": 0}

    for u in urls:
        try:
            parsed = urlparse(u if "://" in u else "https://" + u)
        except ValueError:
            out.append(u)
            continue

        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""

        if netloc.startswith("www.") and rng.random() < 0.50:
            netloc = netloc[4:]
            stats["www_removed"] += 1
            if rng.random() < 0.25:
                sub = str(rng.choice(_SUBDOMAIN_POOL))
                netloc = f"{sub}.{netloc}"
                stats["subdomain_varied"] += 1

        if scheme == "https" and rng.random() < 0.15:
            scheme = "http"
            stats["scheme_to_http"] += 1

        if path in ("", "/") and rng.random() < 0.68:
            path, q_gen = _generate_safe_path_and_query(rng)
            stats["safe_path_added"] += 1
            if not query and q_gen:
                query = q_gen

        rebuilt = urlunparse((scheme, netloc, path, "", query, ""))
        out.append(rebuilt)

    return out, stats


def _deartifact_phish_urls(urls: list[str], rng: np.random.Generator) -> tuple[list[str], dict[str, int]]:
    """Выровнять распределение путей на части фишинговых URL без путей."""
    out: list[str] = []
    stats = {"phish_path_added": 0, "trailing_slash_normalized": 0}

    for u in urls:
        try:
            parsed = urlparse(u if "://" in u else "http://" + u)
        except ValueError:
            out.append(u)
            continue

        scheme = (parsed.scheme or "http").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""

        if path in ("", "/"):
            if rng.random() < 0.52:
                path, q_gen = _generate_phish_path_and_query(rng)
                stats["phish_path_added"] += 1
                if not query and q_gen:
                    query = q_gen
            elif path == "/" and rng.random() < 0.85:
                path = ""
                stats["trailing_slash_normalized"] += 1

        rebuilt = urlunparse((scheme, netloc, path, "", query, ""))
        out.append(rebuilt)

    return out, stats


def _build_synthetic_url_samples(rng: np.random.Generator) -> list[tuple[str, int]]:
    """Сгенерировать синтетические примеры для редких структурных паттернов URL."""
    samples: list[tuple[str, int]] = []

    # 1. Безопасные URL на известных брендах и синтетических доменах с разнообразными путями
    safe_tlds = ("kz", "com", "org", "net", "edu", "gov.kz", "co.uk", "de", "ru")
    for idx in range(500):
        tld = safe_tlds[idx % len(safe_tlds)]
        prefix = "" if (idx % 2 == 0) else ("www." if idx % 4 == 1 else f"{_SUBDOMAIN_POOL[idx % len(_SUBDOMAIN_POOL)]}.")
        scheme = "https" if (idx % 6 != 0) else "http"
        path, query = _generate_safe_path_and_query(rng) if (idx % 5 != 0) else ("", "")
        q_str = f"?{query}" if query else ""
        domain = f"portal-legit-{idx}.{tld}"
        samples.append((f"{scheme}://{prefix}{domain}{path}{q_str}", 0))

    for brand in BRAND_DOMAINS:
        samples.append((f"https://{brand}", 0))
        samples.append((f"https://www.{brand}", 0))
        for _ in range(4):
            p, q = _generate_safe_path_and_query(rng)
            q_str = f"?{q}" if q else ""
            samples.append((f"https://{brand}{p}{q_str}", 0))
            samples.append((f"https://www.{brand}{p}{q_str}", 0))

    # 2. Фишинговые URL с характерными признаками из Задания 9 (как http, так и https)
    shortener_list = sorted(SHORTENERS)
    susp_tlds = sorted(SUSPICIOUS_TLDS)
    phish_words = (
        "secure-bank-verify",
        "account-update-login",
        "kaspi-login-secure",
        "kaspi-bonus-prize",
        "halyk-security-verify",
        "sberbank-login",
        "post-delivery-pay",
        "egov-payout-bonus",
        "bank-kz-verify",
        "wallet-unlock-confirm",
        "card-suspend-alert",
        "free-prize-claim",
    )

    for idx in range(600):
        pattern = idx % 7
        w = phish_words[idx % len(phish_words)]
        scheme = "https" if (idx % 2 == 0) else "http"
        if pattern == 0:
            # Сокращатель ссылок (с http и https, с путём разной длины)
            sh = shortener_list[idx % len(shortener_list)]
            samples.append((f"{scheme}://{sh}/{w}-{idx}", 1))
            samples.append((f"{scheme}://{sh}/3xYz{idx % 999}", 1))
        elif pattern == 1:
            # IP-адрес вместо домена
            octet3 = (idx // 250) % 250 + 1
            octet4 = (idx % 250) + 1
            port = ":8080" if (idx % 3 == 0) else ""
            samples.append((f"{scheme}://192.0.{octet3}.{octet4}{port}/account-update/login?id={idx}", 1))
        elif pattern == 2:
            # Punycode на разных TLD
            tld = ("test", "example", "com", "ru", "kz")[idx % 5]
            samples.append((f"{scheme}://xn--exmple-{idx}cua.{tld}/secure/login?verify=1", 1))
            samples.append((f"{scheme}://xn--80ak6aa92e-{idx}.{tld}/", 1))
        elif pattern == 3:
            # Подозрительные слова и бренд-спуфинг на .example / .test / .com
            tld = ("example", "test", "com", "org")[idx % 4]
            samples.append((f"{scheme}://{w}-{idx}.{tld}/login", 1))
        elif pattern == 4:
            # Подозрительный TLD + бренд-спуфинг (с путём и без пути)
            stld = susp_tlds[idx % len(susp_tlds)]
            samples.append((f"{scheme}://{w}-{idx}{stld}/verify/card", 1))
            samples.append((f"{scheme}://{w}-{idx}{stld}", 1))
        elif pattern == 5:
            # Длинные цепочки поддоменов и нестандартный порт
            tld = ("com", "net", "org", "example", "test")[idx % 5]
            samples.append((f"{scheme}://login.verify.account.secure-{idx}.{tld}:8443/update", 1))
            samples.append((f"{scheme}://auth.signin.bank.portal-{idx}.{tld}/confirm", 1))
        else:
            # Символ @, нестандартный порт и двойной слэш в пути
            stld = susp_tlds[idx % len(susp_tlds)]
            samples.append((f"{scheme}://user@evil-login-{idx}{stld}:8080//confirm?token={idx}", 1))

    return samples


def _describe_dist(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.25, 0.50, 0.75, 0.90, 0.95]).to_dict()
    return {
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "q25": round(float(q[0.25]), 4),
        "q50": round(float(q[0.50]), 4),
        "q75": round(float(q[0.75]), 4),
        "q90": round(float(q[0.90]), 4),
        "q95": round(float(q[0.95]), 4),
        "max": round(float(series.max()), 4),
        "nonzero_share": round(float((series > 0).mean()), 4),
    }


def _compute_raw_distributions(df_safe_raw: pd.DataFrame, df_phish_raw: pd.DataFrame) -> dict[str, Any]:
    """Посчитать распределения path_length и num_digits_url ДО устранения артефактов."""
    def _quick_stats(urls: pd.Series) -> tuple[pd.Series, pd.Series]:
        path_lens = urls.map(
            lambda u: len(urlparse(u if "://" in str(u) else "http://" + str(u)).path or "")
        )
        digits = urls.map(lambda u: sum(ch.isdigit() for ch in str(u)))
        return path_lens, digits

    p0, d0 = _quick_stats(df_safe_raw["url"])
    p1, d1 = _quick_stats(df_phish_raw["url"])
    return {
        "path_length": {"safe_0": _describe_dist(p0), "dangerous_1": _describe_dist(p1)},
        "num_digits_url": {"safe_0": _describe_dist(d0), "dangerous_1": _describe_dist(d1)},
    }


def prepare_url_dataset(
    raw_path: Path | str = DEFAULT_RAW_PATH,
    parquet_path: Path | str = DEFAULT_PARQUET_PATH,
    split_path: Path | str = DEFAULT_SPLIT_PATH,
    sample_size: int = 60_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Подготовить датасет URL, выполнить разбиение по доменам и извлечь 22 признака."""
    raw_path = Path(raw_path)
    parquet_path = Path(parquet_path)
    split_path = Path(split_path)

    df_raw = pd.read_csv(
        raw_path,
        usecols=["URL", "label"],
        encoding="utf-8-sig",
    )
    initial_rows = len(df_raw)

    df_raw = df_raw.dropna(subset=["URL", "label"]).copy()
    df_raw["url"] = df_raw["URL"].astype(str).map(_normalize_url_case)
    df_raw = df_raw[df_raw["url"] != ""].copy()

    # Инверсия метки: в PhiUSIIL 1 = legitimate, 0 = phishing ->
    # в проекте 0 = SAFE, 1 = DANGEROUS
    df_raw["label"] = 1 - df_raw["label"].astype(int)
    df_raw = df_raw[df_raw["label"].isin([0, 1])].copy()

    before_dedup = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    duplicates_removed = before_dedup - len(df_raw)

    # Стратифицированная подвыборка 60 000 записей (30 000 SAFE + 30 000 DANGEROUS)
    half = sample_size // 2
    df_safe = df_raw[df_raw["label"] == 0].sample(n=half, random_state=random_state).copy()
    df_phish = df_raw[df_raw["label"] == 1].sample(n=half, random_state=random_state).copy()

    # Статистика распределений ДО устранения артефактов
    distributions_before = _compute_raw_distributions(df_safe, df_phish)

    rng = np.random.default_rng(random_state)
    deartifacted_safe_urls, safe_artifact_stats = _deartifact_safe_urls(
        df_safe["url"].tolist(), rng
    )
    df_safe["url"] = deartifacted_safe_urls
    df_safe["is_synthetic"] = 0

    deartifacted_phish_urls, phish_artifact_stats = _deartifact_phish_urls(
        df_phish["url"].tolist(), rng
    )
    df_phish["url"] = deartifacted_phish_urls
    df_phish["is_synthetic"] = 0

    artifact_stats = {**safe_artifact_stats, **phish_artifact_stats}

    # Синтетические примеры помечаются явно is_synthetic = 1
    synth_rows = _build_synthetic_url_samples(rng)
    df_synth = pd.DataFrame(synth_rows, columns=["url", "label"])
    df_synth["is_synthetic"] = 1

    df_all = pd.concat(
        [
            df_safe[["url", "label", "is_synthetic"]],
            df_phish[["url", "label", "is_synthetic"]],
            df_synth[["url", "label", "is_synthetic"]],
        ],
        ignore_index=True,
    )

    valid_mask = df_all["url"].map(lambda u: validate_url(u)[0])
    invalid_removed = int((~valid_mask).sum())
    df_all = df_all[valid_mask].drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    df_all["domain"] = df_all["url"].map(_extract_domain_group)

    # Разбиение 70 / 15 / 15 по группам регистрируемого домена (GroupShuffleSplit)
    gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=random_state)
    train_idx, temp_idx = next(gss_train.split(df_all, df_all["label"], groups=df_all["domain"]))

    df_temp = df_all.iloc[temp_idx]
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=random_state)
    val_rel_idx, test_rel_idx = next(
        gss_val.split(df_temp, df_temp["label"], groups=df_temp["domain"])
    )
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    split_col = np.empty(len(df_all), dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"
    df_all["split"] = split_col

    train_domains = set(df_all.iloc[train_idx]["domain"])
    val_domains = set(df_all.iloc[val_idx]["domain"])
    test_domains = set(df_all.iloc[test_idx]["domain"])
    assert not (train_domains & val_domains), "Domain leakage between train and val!"
    assert not (train_domains & test_domains), "Domain leakage between train and test!"
    assert not (val_domains & test_domains), "Domain leakage between val and test!"

    feature_dicts = [extract_url_features(u) for u in df_all["url"]]
    df_features = pd.DataFrame(feature_dicts, columns=URL_FEATURE_NAMES)

    result_df = pd.concat(
        [df_all[["url", "domain", "label", "split", "is_synthetic"]], df_features],
        axis=1,
    )

    # Статистика распределений ПОСЛЕ устранения артефактов (на не-синтетических и всех строках)
    non_synth = result_df[result_df["is_synthetic"] == 0]
    distributions_after = {
        "path_length": {
            "safe_0": _describe_dist(non_synth[non_synth["label"] == 0]["path_length"]),
            "dangerous_1": _describe_dist(non_synth[non_synth["label"] == 1]["path_length"]),
        },
        "num_digits_url": {
            "safe_0": _describe_dist(non_synth[non_synth["label"] == 0]["num_digits_url"]),
            "dangerous_1": _describe_dist(non_synth[non_synth["label"] == 1]["num_digits_url"]),
        },
    }

    synthetic_breakdown = {}
    for s_name in ("train", "val", "test"):
        sub = result_df[result_df["split"] == s_name]
        n_synth = int((sub["is_synthetic"] == 1).sum())
        n_total = int(len(sub))
        synthetic_breakdown[s_name] = {
            "total": n_total,
            "non_synthetic": int(n_total - n_synth),
            "synthetic": n_synth,
            "synthetic_share": round(n_synth / max(1, n_total), 4),
        }

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(parquet_path, index=False)

    split_info: dict[str, Any] = {
        "dataset": "PhiUSIIL Phishing URL Dataset (UCI ML Repository, ID=967)",
        "raw_initial_rows": int(initial_rows),
        "duplicates_removed": int(duplicates_removed),
        "invalid_removed": int(invalid_removed),
        "final_rows": int(len(result_df)),
        "class_counts": {
            "safe_0": int((result_df["label"] == 0).sum()),
            "dangerous_1": int((result_df["label"] == 1).sum()),
        },
        "artifact_mitigation": artifact_stats,
        "distributions_before": distributions_before,
        "distributions_after": distributions_after,
        "synthetic_breakdown": synthetic_breakdown,
        "split_counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "unique_domains": {
            "train": int(len(train_domains)),
            "val": int(len(val_domains)),
            "test": int(len(test_domains)),
            "overlap_train_val": int(len(train_domains & val_domains)),
            "overlap_train_test": int(len(train_domains & test_domains)),
            "overlap_val_test": int(len(val_domains & test_domains)),
        },
        "indices": {
            "train": [int(i) for i in train_idx.tolist()],
            "val": [int(i) for i in val_idx.tolist()],
            "test": [int(i) for i in test_idx.tolist()],
        },
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    return result_df, split_info


if __name__ == "__main__":
    df_out, info = prepare_url_dataset()
    print(
        f"URL dataset prepared: {len(df_out)} rows, "
        f"train={info['split_counts']['train']}, "
        f"val={info['split_counts']['val']}, "
        f"test={info['split_counts']['test']}, "
        f"test synthetic share={info['synthetic_breakdown']['test']['synthetic_share']:.2%}"
    )
