"""Загрузка, очистка, устранение артефактов разметки и построение признаков URL (Задание 10).

Модуль читает открытый датасет PhiUSIIL Phishing URL Dataset, приводит метки к
бинарному стандарту проекта (0 — безопасный/SAFE, 1 — фишинг/DANGEROUS),
устраняет структурные артефакты разметки PhiUSIIL (где все легитимные URL имеют
вид ``https://www.<domain>`` без пути), выполняет групповое разбиение
70/15/15 по регистрируемому домену (``GroupShuffleSplit``) и извлекает 22
признака функцией :func:`ml.url_features.extract_url_features`.
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

_SAFE_PATHS = (
    "/about",
    "/contact",
    "/news/article-12",
    "/catalog/item",
    "/ru/help",
    "/docs/index.html",
    "/profile",
    "/faq",
    "/shop",
    "/services",
    "/info",
    "/company",
    "/support",
    "/terms",
    "/privacy",
    "/ru/cards",
    "/ru/tracking",
    "/credits",
    "/personal",
    "/wiki/Main_Page",
)

_SAFE_QUERIES = (
    "",
    "",
    "",
    "",
    "lang=ru",
    "page=1",
    "q=python",
    "ref=nav",
)


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


def _deartifact_safe_urls(urls: list[str], rng: np.random.Generator) -> tuple[list[str], dict[str, int]]:
    """Устранить артефакты разметки PhiUSIIL на безопасных URL.

    В исходном наборе PhiUSIIL 100% безопасных URL имеют вид
    ``https://www.<domain>`` с пустым путём (``path_length == 0``,
    ``num_subdomains >= 1``, ``is_https == 1``). Если обучить модель без
    коррекции, она выучит ложное правило «любой путь или отсутствие www —
    это фишинг».

    Для устранения артефакта у части безопасных URL:
    1. Убирается префикс ``www.`` (~45%).
    2. Схема ``https://`` заменяется на ``http://`` (~20%).
    3. Добавляется типичный безопасный путь из ``_SAFE_PATHS`` (~55%) и
       опциональный безопасный query (~15% от путей).
    """
    out: list[str] = []
    stats = {"www_removed": 0, "scheme_to_http": 0, "safe_path_added": 0}

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

        if netloc.startswith("www.") and rng.random() < 0.45:
            netloc = netloc[4:]
            stats["www_removed"] += 1

        if scheme == "https" and rng.random() < 0.20:
            scheme = "http"
            stats["scheme_to_http"] += 1

        if path in ("", "/") and rng.random() < 0.55:
            path = str(rng.choice(_SAFE_PATHS))
            stats["safe_path_added"] += 1
            if not query and rng.random() < 0.20:
                q = str(rng.choice(_SAFE_QUERIES))
                if q:
                    query = q

        rebuilt = urlunparse((scheme, netloc, path, "", query, ""))
        out.append(rebuilt)

    return out, stats


def _build_synthetic_url_samples(rng: np.random.Generator) -> list[tuple[str, int]]:
    """Сгенерировать синтетические безопасные и фишинговые URL на отдельных доменах.

    Это гарантирует, что модель корректно учитывает:
    - легитимные казахстанские и мировые домены с путями и без ``www.``;
    - сокращатели ссылок (``bit.ly``, ``tinyurl.com``, ``t.co``, ``clck.ru`` и др.);
    - IP-адреса вместо домена, Punycode (``xn--``), бренд-спуфинг, подозрительные
      слова и подозрительные TLD (``.xyz``, ``.top``, ``.click``, ``.icu``).
    """
    samples: list[tuple[str, int]] = []

    # 1. Безопасные URL на известных брендах и синтетических доменах
    safe_tlds = ("kz", "com", "org", "net", "edu", "gov.kz", "co.uk", "de")
    for idx in range(600):
        tld = safe_tlds[idx % len(safe_tlds)]
        use_www = "www." if (idx % 2 == 0) else ""
        scheme = "https" if (idx % 5 != 0) else "http"
        path = _SAFE_PATHS[idx % len(_SAFE_PATHS)] if (idx % 3 != 0) else ""
        query = "?q=python" if (idx % 11 == 0 and path) else ""
        domain = f"portal-legit-{idx}.{tld}"
        samples.append((f"{scheme}://{use_www}{domain}{path}{query}", 0))

    # Вариации легитимных брендов с путями и без www
    for brand in BRAND_DOMAINS:
        samples.append((f"https://{brand}", 0))
        samples.append((f"https://www.{brand}", 0))
        for p in ("/about", "/ru/cards", "/ru/tracking", "/credits", "/personal", "/search?q=python"):
            samples.append((f"https://{brand}{p}", 0))
            samples.append((f"https://www.{brand}{p}", 0))

    # 2. Фишинговые URL с характерными признаками из Задания 9
    shortener_list = sorted(SHORTENERS)
    susp_tlds = sorted(SUSPICIOUS_TLDS)
    phish_words = (
        "secure-bank-verify",
        "account-update-login",
        "kaspi-bonus-prize",
        "halyk-security-verify",
        "post-delivery-pay",
        "egov-payout-bonus",
        "bank-kz-verify",
        "wallet-unlock-confirm",
        "card-suspend-alert",
        "free-prize-claim",
    )

    for idx in range(600):
        pattern = idx % 6
        w = phish_words[idx % len(phish_words)]
        if pattern == 0:
            # Сокращатель ссылок
            sh = shortener_list[idx % len(shortener_list)]
            samples.append((f"http://{sh}/{w}-{idx}", 1))
            samples.append((f"{sh}/prize-claim-{idx}", 1))
        elif pattern == 1:
            # IP-адрес вместо домена
            octet3 = (idx // 250) % 250 + 1
            octet4 = (idx % 250) + 1
            port = ":8080" if (idx % 3 == 0) else ""
            samples.append((f"http://192.0.{octet3}.{octet4}{port}/account-update/login?id={idx}", 1))
        elif pattern == 2:
            # Punycode и резервные домены .test / .example
            tld = "test" if (idx % 2 == 0) else "example"
            samples.append((f"http://xn--exmple-{idx}cua.{tld}/secure/login?verify=1", 1))
        elif pattern == 3:
            # Подозрительные слова на .example / .test
            tld = "example" if (idx % 2 == 0) else "test"
            samples.append((f"http://{w}-{idx}.{tld}/login", 1))
        elif pattern == 4:
            # Подозрительный TLD + бренд-спуфинг
            stld = susp_tlds[idx % len(susp_tlds)]
            samples.append((f"http://{w}-{idx}{stld}/verify/card", 1))
        else:
            # Символ @, нестандартный порт и двойной слэш в пути
            stld = susp_tlds[idx % len(susp_tlds)]
            samples.append((f"http://user@evil-login-{idx}{stld}:8080//confirm?token={idx}", 1))

    return samples


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
    df_safe = df_raw[df_raw["label"] == 0].sample(n=half, random_state=random_state)
    df_phish = df_raw[df_raw["label"] == 1].sample(n=half, random_state=random_state)

    rng = np.random.default_rng(random_state)
    deartifacted_safe_urls, artifact_stats = _deartifact_safe_urls(
        df_safe["url"].tolist(), rng
    )
    df_safe = df_safe.copy()
    df_safe["url"] = deartifacted_safe_urls

    # Добавляем синтетические примеры для покрытия граничных фишинговых и безопасных паттернов
    synth_rows = _build_synthetic_url_samples(rng)
    df_synth = pd.DataFrame(synth_rows, columns=["url", "label"])

    df_all = pd.concat(
        [df_safe[["url", "label"]], df_phish[["url", "label"]], df_synth],
        ignore_index=True,
    )

    # Оставляем только валидные URL
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

    # Проверка отсутствия пересечения доменов
    train_domains = set(df_all.iloc[train_idx]["domain"])
    val_domains = set(df_all.iloc[val_idx]["domain"])
    test_domains = set(df_all.iloc[test_idx]["domain"])
    assert not (train_domains & val_domains), "Domain leakage between train and val!"
    assert not (train_domains & test_domains), "Domain leakage between train and test!"
    assert not (val_domains & test_domains), "Domain leakage between val and test!"

    # Извлечение 22 признаков функцией из Задания 9
    feature_dicts = [extract_url_features(u) for u in df_all["url"]]
    df_features = pd.DataFrame(feature_dicts, columns=URL_FEATURE_NAMES)

    result_df = pd.concat([df_all[["url", "domain", "label", "split"]], df_features], axis=1)

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
        f"domain overlaps=0"
    )
