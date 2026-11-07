"""Регрессионные тесты обученной URL-модели (Задание 11).

Проверяет через :meth:`ml.predictor.ModelPredictor.get_instance` отсутствие
ложных срабатываний ``DANGEROUS`` на реальных безопасных URL (с путями,
числовыми сегментами, параметрами запроса и доменами известных банков) и
высокую полноту обнаружения (``DANGEROUS`` / ``SUSPICIOUS``) на 15+
фишинговых URL различных структурных типов.
"""

from __future__ import annotations

import pytest

from ml.predictor import ModelPredictor

REAL_SAFE_URLS: list[str] = [
    "https://github.com/python/cpython",
    "https://docs.python.org/3/",
    "https://mail.google.com/mail/u/0/",
    "https://www.sberbank.ru/ru/person",
    "https://kaspi.kz",
    "https://egov.kz",
    "https://ru.wikipedia.org/wiki/Python",
    "https://habr.com/ru/articles/123456/",
    "https://www.google.com/search?q=weather",
    "https://www.example.com/about",
    "https://halykbank.kz/ru/cards",
    "https://kazpost.kz/ru/tracking",
    "https://www.python.org/downloads/release/python-3120/",
    "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
]

PHISHING_URLS: list[str] = [
    "http://192.0.2.15/account-update",
    "http://xn--exmplecua.test/secure/login?verify=1",
    "http://user@evil-login-pay.click:8080//confirm",
    "bit.ly/prize-claim-test",
    "https://bit.ly/3xYz987",
    "https://kaspi-login-secure.xyz/verify",
    "http://sberbank-login.xyz",
    "https://login.verify.account.security.example.com/update",
    "http://example.com:8080/login/verify",
    "http://secure-bank-verify.example/login",
    "http://bank-kz-verify.test",
    "http://post-delivery-pay.example",
    "http://kaspi-bonus-prize.xyz/claim",
    "http://halyk-security-update.top/verify/card",
    "http://egov-payout-bonus.icu/wallet/unlock",
]


@pytest.fixture(scope="module")
def predictor() -> ModelPredictor:
    ModelPredictor.reset_instance()
    inst = ModelPredictor.get_instance()
    assert inst.url_loaded, "URL model must be loaded for regression tests"
    return inst


def test_safe_urls_count_at_least_12() -> None:
    assert len(REAL_SAFE_URLS) >= 12


def test_phishing_urls_count_at_least_15() -> None:
    assert len(PHISHING_URLS) >= 15


@pytest.mark.parametrize("url", REAL_SAFE_URLS)
def test_real_safe_url_not_dangerous(predictor: ModelPredictor, url: str) -> None:
    result = predictor.predict_url(url)
    assert result["category"] != "DANGEROUS", (
        f"False positive DANGEROUS on safe URL {url}: "
        f"p_url={result['p_url']:.4f}, category={result['category']}"
    )
    assert result["category"] == "SAFE", (
        f"Expected SAFE on {url}, got {result['category']} (p_url={result['p_url']:.4f})"
    )


def test_phishing_urls_detected(predictor: ModelPredictor) -> None:
    detected = 0
    details: list[str] = []
    for url in PHISHING_URLS:
        result = predictor.predict_url(url)
        cat = result["category"]
        details.append(f"{url} -> p_url={result['p_url']:.4f} ({cat})")
        if cat in ("DANGEROUS", "SUSPICIOUS"):
            detected += 1

    assert detected >= 14, (
        f"Expected at least 14/{len(PHISHING_URLS)} phishing URLs to be "
        f"DANGEROUS/SUSPICIOUS, got {detected}:\n" + "\n".join(details)
    )
