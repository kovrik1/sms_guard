"""Модульные тесты для ml/url_features.py.

Все тесты полностью офлайновые: используются только синтетические URL
(зарезервированные зоны .example/.test, example.com, IP-адреса и т.п.).
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml import url_features  # noqa: E402
from ml.url_features import (  # noqa: E402
    BRAND_DOMAINS,
    SHORTENERS,
    SUSPICIOUS_TLDS,
    URL_FEATURE_NAMES,
    explain_url,
    extract_url_features,
    validate_url,
)


# --------------------------------------------------------------------------- #
# Список признаков
# --------------------------------------------------------------------------- #

def test_feature_names_count_is_22():
    assert len(URL_FEATURE_NAMES) == 22


def test_feature_names_exact_order():
    assert URL_FEATURE_NAMES == [
        'url_length', 'domain_length', 'path_length', 'num_dots',
        'num_subdomains', 'num_hyphens_domain', 'num_digits_url',
        'digit_ratio_domain', 'num_special_chars', 'has_ip_address',
        'is_https', 'has_non_standard_port', 'has_at_symbol',
        'has_double_slash_in_path', 'is_shortener', 'num_suspicious_words',
        'has_punycode', 'has_mixed_scripts', 'brand_similarity',
        'domain_entropy', 'is_suspicious_tld', 'num_query_params',
    ]
    assert len(set(URL_FEATURE_NAMES)) == 22


def test_brand_and_shortener_lists_are_large_enough():
    assert len(BRAND_DOMAINS) >= 22
    assert len(SHORTENERS) >= 18
    assert len(SUSPICIOUS_TLDS) >= 10


# --------------------------------------------------------------------------- #
# validate_url
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('url', [
    'https://example.com',
    'http://example.com/path?x=1',
    'example.com',
    'www.example.com:8080/login',
    'https://sub.example.com/a/b/c',
    'http://192.0.2.10/admin',
    'http://[2001:db8::1]:8080/',
])
def test_validate_url_accepts_valid(url):
    is_valid, normalized, error = validate_url(url)
    assert is_valid is True
    assert error == ''
    assert normalized.startswith(('http://', 'https://'))


def test_validate_url_adds_scheme_for_bare_domain():
    is_valid, normalized, _ = validate_url('example.com/path')
    assert is_valid is True
    assert normalized == 'http://example.com/path'


@pytest.mark.parametrize('url', [
    'ftp://example.com/file',
    'javascript:alert(1)',
    'data:text/html;base64,PHNjcmlwdD4=',
    'file:///etc/passwd',
    'ws://example.com/socket',
])
def test_validate_url_rejects_foreign_schemes(url):
    is_valid, normalized, error = validate_url(url)
    assert is_valid is False
    assert normalized == ''
    assert error


@pytest.mark.parametrize('url', [None, '', '   ', 123, [], b'http://example.com'])
def test_validate_url_rejects_empty_and_non_string(url):
    is_valid, normalized, error = validate_url(url)
    assert is_valid is False
    assert normalized == ''
    assert error


def test_validate_url_rejects_whitespace_inside():
    is_valid, _, error = validate_url('http://exa mple.com/login')
    assert is_valid is False
    assert 'пробел' in error.lower()


def test_validate_url_rejects_too_long():
    url = 'http://example.com/' + 'a' * 2100
    is_valid, _, error = validate_url(url)
    assert is_valid is False
    assert 'длин' in error.lower()


@pytest.mark.parametrize('url', [
    'http://',
    'http://example',          # нет точки
    'http://example..com',     # пустая часть
    'http://-bad.example',     # метка начинается с дефиса
    'http://example.com:99999/',
])
def test_validate_url_rejects_bad_hosts(url):
    is_valid, normalized, error = validate_url(url)
    assert is_valid is False
    assert normalized == ''
    assert error


def test_validate_url_accepts_ipv4_and_ipv6():
    ok4, norm4, _ = validate_url('http://192.0.2.1/login')
    ok6, norm6, _ = validate_url('https://[2001:db8::2]/path')
    assert ok4 and ok6
    assert '192.0.2.1' in norm4
    assert '[2001:db8::2]' in norm6


# --------------------------------------------------------------------------- #
# extract_url_features: базовая структура
# --------------------------------------------------------------------------- #

def test_extract_keys_order_matches_feature_names():
    features = extract_url_features('https://example.com/a?b=1')
    assert list(features.keys()) == URL_FEATURE_NAMES


def test_extract_values_are_numeric():
    features = extract_url_features('https://example.com/a?b=1')
    assert all(isinstance(v, (int, float)) for v in features.values())


@pytest.mark.parametrize('bad', [
    None, '', '   ', 'not a url', 'javascript:alert(1)', 'http://', 42,
    'ftp://example.com/x', 'http://example..com',
])
def test_invalid_input_returns_all_zero_features(bad):
    features = extract_url_features(bad)
    assert list(features.keys()) == URL_FEATURE_NAMES
    assert all(v == 0 for v in features.values())


def test_basic_lengths_and_counts():
    url = 'https://www.example.com/path/to/page?x=1'
    f = extract_url_features(url)
    assert f['url_length'] == len(url)
    assert f['domain_length'] == len('www.example.com')
    assert f['path_length'] == len('/path/to/page')
    assert f['num_dots'] == 2
    assert f['num_digits_url'] == 1


# --------------------------------------------------------------------------- #
# Отдельные признаки
# --------------------------------------------------------------------------- #

def test_https_flag():
    assert extract_url_features('https://example.com/')['is_https'] == 1
    assert extract_url_features('http://example.com/')['is_https'] == 0
    # без схемы HTTPS не «явный»
    assert extract_url_features('example.com/')['is_https'] == 0


def test_ip_address_flag_and_no_subdomains():
    f = extract_url_features('http://192.0.2.10/login')
    assert f['has_ip_address'] == 1
    assert f['num_subdomains'] == 0
    assert f['domain_length'] == len('192.0.2.10')


def test_ipv6_features():
    f = extract_url_features('http://[2001:db8::1]:8080/admin')
    assert f['has_ip_address'] == 1
    assert f['has_non_standard_port'] == 1
    assert f['num_subdomains'] == 0


def test_non_standard_port():
    assert extract_url_features('http://example.com:8080/x')['has_non_standard_port'] == 1
    assert extract_url_features('https://example.com:443/x')['has_non_standard_port'] == 0
    assert extract_url_features('http://example.com:80/x')['has_non_standard_port'] == 0
    assert extract_url_features('https://example.com/x')['has_non_standard_port'] == 0


def test_at_symbol():
    f = extract_url_features('http://user@evil.example/path')
    assert f['has_at_symbol'] == 1
    assert extract_url_features('http://evil.example/path')['has_at_symbol'] == 0


def test_double_slash_in_path():
    assert extract_url_features('http://example.com//admin')['has_double_slash_in_path'] == 1
    assert extract_url_features('http://example.com/admin')['has_double_slash_in_path'] == 0


@pytest.mark.parametrize('url', [
    'https://bit.ly/3abc',
    'http://tinyurl.com/xyz',
    'https://t.co/abc',
    'https://clck.ru/abc',
    'https://www.bit.ly/abc',
])
def test_shorteners_detected(url):
    assert extract_url_features(url)['is_shortener'] == 1


def test_regular_domain_is_not_shortener():
    assert extract_url_features('https://example.com/abc')['is_shortener'] == 0


def test_punycode_detected():
    f = extract_url_features('http://xn--80ak6aa92e.example/')
    assert f['has_punycode'] == 1
    assert extract_url_features('http://example.com/')['has_punycode'] == 0


def test_mixed_scripts_detected():
    # «а» — кириллическая, остальное латиница
    f = extract_url_features('https://\u0430pple.example/login')
    assert f['has_mixed_scripts'] == 1
    assert extract_url_features('https://apple.example/login')['has_mixed_scripts'] == 0


def test_suspicious_tld():
    assert extract_url_features('https://secure-login.xyz/')['is_suspicious_tld'] == 1
    assert extract_url_features('https://example.top/')['is_suspicious_tld'] == 1
    assert extract_url_features('https://example.com/')['is_suspicious_tld'] == 0


def test_subdomains_hyphens_and_digit_ratio():
    f = extract_url_features('https://a.b.c.example-shop2.com/x')
    assert f['num_subdomains'] == 3
    assert f['num_hyphens_domain'] == 1
    assert 0.0 < f['digit_ratio_domain'] < 1.0
    assert extract_url_features('https://example.com/')['digit_ratio_domain'] == 0.0


def test_suspicious_words_counted():
    f = extract_url_features('https://secure-login-bank.example/verify/account')
    # secure, login, bank, verify, account
    assert f['num_suspicious_words'] == 5
    assert extract_url_features('https://example.com/about')['num_suspicious_words'] == 0


def test_query_params_counted():
    f = extract_url_features('https://example.com/search?a=1&b=2&c=')
    assert f['num_query_params'] == 3
    assert extract_url_features('https://example.com/search')['num_query_params'] == 0


def test_special_chars_counted():
    f = extract_url_features('https://example.com/a~b%c=d&e_f?g')
    assert f['num_special_chars'] == 6


def test_domain_entropy():
    clean = extract_url_features('https://example.com/')['domain_entropy']
    randomish = extract_url_features('https://a8f3k2m9q1z7.example/')['domain_entropy']
    assert 0.0 <= clean <= 4.5
    assert randomish > clean


# --------------------------------------------------------------------------- #
# brand_similarity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('url', [
    'https://kaspi.kz/login',
    'https://my.kaspi.kz/pay',
    'https://halykbank.kz/',
    'https://www.google.com/search',
    'https://paypal.com/signin',
    'https://example.com/',
])
def test_legit_brand_domain_not_flagged(url):
    assert extract_url_features(url)['brand_similarity'] == 0


@pytest.mark.parametrize('url', [
    'https://kaspi-verify.example/login',
    'https://kaspi.xyz/',
    'https://login.kaspi.com.test/',
    'https://kasp1.example/',
    'https://micros0ft-login.example/',
    'https://paypa1.example/',
])
def test_brand_spoofing_flagged(url):
    assert extract_url_features(url)['brand_similarity'] == 1


# --------------------------------------------------------------------------- #
# explain_url
# --------------------------------------------------------------------------- #

def test_explain_url_reports_dangerous_features():
    features = extract_url_features('http://192.0.2.10:8080/login//verify?x=1')
    reasons = explain_url(features)
    assert reasons
    joined = ' | '.join(reasons)
    assert 'IP-адрес' in joined
    assert 'порт' in joined.lower()
    assert 'HTTPS' in joined
    assert '//' in joined


def test_explain_url_clean_url_has_no_reasons():
    assert explain_url(extract_url_features('https://example.com/path')) == []


def test_explain_url_handles_empty_or_foreign_input():
    assert explain_url({}) == []
    assert explain_url(None) == []
    assert explain_url({'has_ip_address': 'yes', 'is_https': 0})


def test_explain_url_entropy_and_length_thresholds():
    reasons = explain_url({'domain_entropy': 3.9, 'url_length': 150})
    joined = ' | '.join(reasons)
    assert 'энтропия' in joined.lower()
    assert 'длинный' in joined.lower()


# --------------------------------------------------------------------------- #
# Отсутствие сетевой активности
# --------------------------------------------------------------------------- #

def test_module_has_no_network_imports():
    source = inspect.getsource(url_features)
    forbidden = (
        'import requests', 'import socket', 'from socket',
        'import urllib.request', 'from urllib.request', 'urllib.request',
        'from urllib import request', 'import http.client',
        'http.client', 'httpx', 'aiohttp', 'urlopen', 'getaddrinfo',
    )
    for token in forbidden:
        assert token not in source, f'найден сетевой код: {token}'


def test_tldextract_configured_offline():
    source = inspect.getsource(url_features)
    assert 'suffix_list_urls=()' in source
    assert 'cache_dir=None' in source


def test_extract_does_not_create_network_sockets(monkeypatch):
    """Любая попытка открыть соединение должна ломать тест."""

    def _boom(*args, **kwargs):
        raise AssertionError('сетевой вызов в extract_url_features')

    monkeypatch.setattr('socket.socket.connect', _boom, raising=False)
    monkeypatch.setattr('socket.create_connection', _boom, raising=False)
    monkeypatch.setattr('socket.getaddrinfo', _boom, raising=False)

    features = extract_url_features('https://secure-login.xyz/path?a=1')
    assert len(features) == 22
