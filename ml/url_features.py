"""Извлечение признаков URL для детекции фишинговых ссылок.

Модуль работает ИСКЛЮЧИТЕЛЬНО со строковым разбором: никаких сетевых
запросов (HTTP-клиенты, DNS-резолвинг, открытие соединений) не выполняется.
tldextract создаётся с пустым списком источников суффиксов и без кэша на
диске, то есть использует только встроенный офлайн-снимок Public Suffix List.

Публичный API:
    URL_FEATURE_NAMES  - фиксированный список из 22 признаков (порядок важен)
    validate_url(url)  -> (is_valid, normalized_url, error_message)
    extract_url_features(url) -> dict из 22 числовых признаков
    explain_url(features) -> список человекочитаемых причин (на русском)
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

import tldextract

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

URL_FEATURE_NAMES: List[str] = [
    'url_length', 'domain_length', 'path_length', 'num_dots', 'num_subdomains',
    'num_hyphens_domain', 'num_digits_url', 'digit_ratio_domain', 'num_special_chars',
    'has_ip_address', 'is_https', 'has_non_standard_port', 'has_at_symbol',
    'has_double_slash_in_path', 'is_shortener', 'num_suspicious_words', 'has_punycode',
    'has_mixed_scripts', 'brand_similarity', 'domain_entropy', 'is_suspicious_tld',
    'num_query_params',
]

MAX_URL_LENGTH = 2048

# Спецсимволы, учитываемые в num_special_chars.
SPECIAL_CHARS = ('@', '~', '%', '=', '&', '_', '?')

# URL-сокращатели (18 сервисов).
SHORTENERS = frozenset({
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'is.gd', 'cutt.ly', 'ow.ly',
    'clck.ru', 'vk.cc', 'rb.gy', 'shorturl.at', 'tiny.cc', 'buff.ly',
    'rebrand.ly', 'bl.ink', 's.id', 'u.to', 'inx.lv',
})

# Подозрительные доменные зоны (с ведущей точкой).
SUSPICIOUS_TLDS = frozenset({
    '.xyz', '.top', '.click', '.zip', '.loan', '.work', '.gq', '.tk', '.ml',
    '.cf', '.ga', '.icu', '.buzz', '.monster', '.rest', '.fit',
})

# Слова-маркеры фишинга.
SUSPICIOUS_WORDS = (
    'login', 'verify', 'secure', 'account', 'update', 'bank', 'confirm',
    'password', 'bonus', 'free', 'prize', 'kaspi', 'halyk', 'signin',
    'wallet', 'unlock', 'suspend', 'pay', 'card', 'egov',
)

# Бренды, которые чаще всего подделывают (>= 22).
BRAND_DOMAINS = (
    'kaspi.kz', 'kaspibank.kz', 'halykbank.kz', 'halyk.kz', 'jusan.kz',
    'bcc.kz', 'forte.kz', 'egov.kz', 'kazpost.kz', 'sberbank.ru', 'sber.ru',
    'tinkoff.ru', 'tbank.ru', 'vtb.ru', 'alfabank.ru', 'gosuslugi.ru',
    'paypal.com', 'apple.com', 'google.com', 'microsoft.com', 'amazon.com',
    'netflix.com', 'binance.com', 'telegram.org', 'whatsapp.com',
    'instagram.com',
)

# Имена брендов без TLD (+ распространённый алиас).
_BRAND_NAMES: Tuple[str, ...] = tuple(sorted(
    {domain.split('.')[0] for domain in BRAND_DOMAINS} | {'halyk'}
))

# Гомоглифы, используемые для маскировки брендов.
_HOMOGLYPHS_TRANSLATION = str.maketrans({
    '0': 'o', '1': 'i', '3': 'e', '5': 's', '@': 'a',
})

_CYRILLIC_RE = re.compile(r'[\u0400-\u04FF]')
_LATIN_RE = re.compile(r'[A-Za-z]')
_SCHEME_RE = re.compile(r'^([A-Za-z][A-Za-z0-9+.\-]*)://')
_FORBIDDEN_SCHEME_RE = re.compile(
    r'^(javascript|data|vbscript|file|ftp|ftps|mailto|tel|about|blob|'
    r'chrome|ws|wss|gopher|ldap|sftp|ssh|smb):',
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r'\s')
_LABEL_RE = re.compile(r'^[A-Za-z0-9_\-\u00a1-\uffff]+$')

# Офлайн-экстрактор: пустой список источников суффиксов, без кэша на диске.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def _zero_features() -> Dict[str, float]:
    """Словарь из всех 22 признаков с нулевыми значениями (в фикс. порядке)."""
    return {name: 0 for name in URL_FEATURE_NAMES}


def _is_ip_address(host: str) -> bool:
    """True, если host — валидный IPv4/IPv6-адрес (без сетевых запросов)."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _is_valid_hostname(host: str) -> bool:
    """Проверка доменного имени: есть точка, части непустые и корректные."""
    if not host or len(host) > 253:
        return False
    if '.' not in host:
        return False
    labels = host.split('.')
    if any(not label for label in labels):
        return False
    for label in labels:
        if len(label) > 63:
            return False
        if label.startswith('-') or label.endswith('-'):
            return False
        if not _LABEL_RE.match(label):
            return False
    return True


def _levenshtein(first: str, second: str) -> int:
    """Расстояние Левенштейна (итеративная реализация)."""
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, char_a in enumerate(first, start=1):
        current = [i]
        for j, char_b in enumerate(second, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _dehomoglyph(text: str) -> str:
    """Замена похожих символов на латинские буквы."""
    result = text.translate(_HOMOGLYPHS_TRANSLATION)
    result = result.replace('rn', 'm')
    result = result.replace('vv', 'w')
    return result


def _shannon_entropy(text: str) -> float:
    """Энтропия Шеннона по символам строки (бит на символ)."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )
    return entropy


def _brand_similarity(hostname: str, registered_domain: str,
                      suffix: str) -> int:
    """Сходство домена с известным брендом: 1 — похож, 0 — нет.

    Точное совпадение (регистрируемый домен или hostname с поддоменами)
    с брендом считается легитимным и даёт 0.
    """
    if not hostname:
        return 0

    host = hostname.lower().rstrip('.')

    # Легитимный домен бренда (включая поддомены: my.kaspi.kz).
    for brand_domain in BRAND_DOMAINS:
        if (host == brand_domain
                or host.endswith('.' + brand_domain)
                or registered_domain == brand_domain):
            return 0

    labels = [part for part in re.split(r'[.\-]', host) if part]

    raw_candidates = set(labels)
    raw_candidates.add(host)
    if registered_domain:
        raw_candidates.add(registered_domain)
        raw_candidates.add(registered_domain.split('.')[0])
    if suffix and host.endswith('.' + suffix):
        raw_candidates.add(host[: -len(suffix) - 1])

    candidates = set()
    for candidate in raw_candidates:
        candidates.add(candidate)
        dehomoglyphed = _dehomoglyph(candidate)
        candidates.add(dehomoglyphed)
        candidates.add(dehomoglyphed.replace('1', 'l'))

    for brand in _BRAND_NAMES:
        if len(brand) >= 4:
            # (б) имя бренда целиком входит в домен/поддомен чужого домена
            if brand in host:
                return 1
            # (а) близкое написание (опечатки/гомоглифы)
            for candidate in candidates:
                if abs(len(candidate) - len(brand)) > 2:
                    continue
                if _levenshtein(candidate, brand) <= 2:
                    return 1
        else:
            # Короткие бренды (vtb, bcc) сравниваем только как целый токен.
            if brand in labels:
                return 1

    return 0


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #

def validate_url(url: str) -> Tuple[bool, str, str]:
    """Проверить формат URL.

    Возвращает (is_valid, normalized_url, error_message). При ошибке
    normalized_url — пустая строка, error_message — причина на русском.
    """
    if not isinstance(url, str):
        return False, '', 'URL должен быть строкой'

    candidate = url.strip()
    if not candidate:
        return False, '', 'Пустой URL'

    if len(candidate) > MAX_URL_LENGTH:
        return False, '', f'URL слишком длинный (максимум {MAX_URL_LENGTH} символов)'

    if _WHITESPACE_RE.search(candidate):
        return False, '', 'URL содержит пробельные символы'

    scheme_match = _SCHEME_RE.match(candidate)
    if scheme_match:
        scheme = scheme_match.group(1).lower()
        if scheme not in ('http', 'https'):
            return False, '', f'Неподдерживаемая схема: {scheme}'
        normalized = candidate
    elif _FORBIDDEN_SCHEME_RE.match(candidate):
        return False, '', 'Неподдерживаемая схема URL'
    elif candidate.startswith('//'):
        normalized = 'http:' + candidate
    else:
        normalized = 'http://' + candidate

    try:
        parsed = urlparse(normalized)
        hostname = parsed.hostname
        _ = parsed.port  # может бросить ValueError на некорректном порте
    except ValueError as exc:
        return False, '', f'Некорректный URL: {exc}'

    if not hostname:
        return False, '', 'URL не содержит хоста'

    host = hostname.lower().rstrip('.')
    if not host:
        return False, '', 'Пустой хост'

    if _is_ip_address(host):
        return True, normalized, ''

    if not _is_valid_hostname(host):
        return False, '', 'Некорректное доменное имя (нужна точка и непустые части)'

    return True, normalized, ''


def extract_url_features(url: str) -> Dict[str, float]:
    """Извлечь 22 признака URL в фиксированном порядке.

    Никогда не бросает исключений: для некорректного ввода (None, '',
    невалидный URL) возвращает словарь из всех признаков с нулями.
    """
    try:
        is_valid, normalized, _ = validate_url(url)
    except Exception:
        return _zero_features()

    if not is_valid:
        return _zero_features()

    try:
        return _features_from_valid_url(url.strip(), normalized)
    except Exception:
        return _zero_features()


def _features_from_valid_url(raw: str, normalized: str) -> Dict[str, float]:
    """Посчитать признаки для уже провалидированного URL."""
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or '').lower()
    path = parsed.path or ''
    query = parsed.query or ''
    lowered = normalized.lower()
    raw_lowered = raw.lower()

    is_ip = _is_ip_address(hostname)

    if is_ip:
        registered_domain = hostname
        subdomains = 0
        suffix = ''
    else:
        extracted = _TLD_EXTRACTOR(hostname)
        if hasattr(extracted, 'top_domain_under_public_suffix'):
            registered_domain = (extracted.top_domain_under_public_suffix or '').lower()
        else:
            registered_domain = (getattr(extracted, 'registered_domain', '') or '').lower()
        subdomains = len([
            part for part in (extracted.subdomain or '').split('.') if part
        ])
        suffix = (extracted.suffix or '').lower()

    try:
        port = parsed.port
    except ValueError:
        port = None

    domain_length = len(hostname)
    digits_in_domain = sum(1 for char in hostname if char.isdigit())
    is_legit_brand = registered_domain in BRAND_DOMAINS and '@' not in normalized
    legit_brand_exempt = (
        {registered_domain.split('.')[0], 'bank', 'card', 'pay', 'kaspi', 'halyk', 'egov', 'bonus', 'wallet'}
        if is_legit_brand
        else set()
    )

    values: Dict[str, float] = {
        'url_length': len(raw) or len(normalized),
        'domain_length': domain_length,
        'path_length': len(path),
        'num_dots': normalized.count('.'),
        'num_subdomains': subdomains,
        'num_hyphens_domain': hostname.count('-'),
        'num_digits_url': sum(1 for char in normalized if char.isdigit()),
        'digit_ratio_domain': (
            round(digits_in_domain / domain_length, 4) if domain_length else 0.0
        ),
        'num_special_chars': sum(normalized.count(char) for char in SPECIAL_CHARS),
        'has_ip_address': 1 if is_ip else 0,
        'is_https': 1 if raw_lowered.startswith('https://') else 0,
        'has_non_standard_port': (
            1 if port is not None and port not in (80, 443) else 0
        ),
        'has_at_symbol': 1 if '@' in normalized else 0,
        'has_double_slash_in_path': 1 if '//' in path else 0,
        'is_shortener': 1 if (
            hostname in SHORTENERS
            or any(hostname.endswith('.' + short) for short in SHORTENERS)
            or registered_domain in SHORTENERS
        ) else 0,
        'num_suspicious_words': sum(
            1 for word in SUSPICIOUS_WORDS
            if word in lowered
            and not (is_legit_brand and (word in legit_brand_exempt or word in registered_domain))
        ),
        'has_punycode': 1 if 'xn--' in hostname or 'xn--' in lowered else 0,
        'has_mixed_scripts': (
            1 if _CYRILLIC_RE.search(hostname) and _LATIN_RE.search(hostname) else 0
        ),
        'brand_similarity': _brand_similarity(
            hostname, registered_domain, suffix
        ) if not is_ip else 0,
        'domain_entropy': round(_shannon_entropy(hostname), 4),
        'is_suspicious_tld': 1 if (
            (suffix and ('.' + suffix) in SUSPICIOUS_TLDS)
            or hostname.endswith(tuple(SUSPICIOUS_TLDS))
        ) else 0,
        'num_query_params': len(parse_qs(query, keep_blank_values=True)),
    }

    return {name: values[name] for name in URL_FEATURE_NAMES}


def explain_url(features: dict) -> List[str]:
    """Сформировать человекочитаемые причины подозрительности (рус.)."""
    reasons: List[str] = []
    if not isinstance(features, dict) or not features:
        return reasons

    def value(name: str, default: float = 0) -> float:
        raw_value = features.get(name, default)
        return raw_value if isinstance(raw_value, (int, float)) else default

    if value('has_ip_address'):
        reasons.append('Вместо домена используется IP-адрес')
    if value('is_shortener'):
        reasons.append('URL сокращён через сервис коротких ссылок')
    if value('has_punycode'):
        reasons.append('Домен содержит Punycode (xn--), что скрывает реальное имя')
    if value('has_mixed_scripts'):
        reasons.append('Смешение кириллицы и латиницы в домене (омоглифная атака)')
    if value('brand_similarity'):
        reasons.append('Домен похож на известный бренд (возможный спуфинг)')
    if value('is_suspicious_tld'):
        reasons.append('Подозрительная доменная зона')
    if value('has_at_symbol'):
        reasons.append('Символ "@" в URL маскирует реальный домен')
    if value('has_non_standard_port'):
        reasons.append('Используется нестандартный сетевой порт')
    if value('has_double_slash_in_path'):
        reasons.append('Двойной слэш "//" в пути URL')
    if 'is_https' in features and not value('is_https', 1):
        reasons.append('Соединение без HTTPS')
    if value('num_subdomains') >= 3:
        reasons.append(
            f'Слишком много поддоменов: {int(value("num_subdomains"))}'
        )
    if value('num_hyphens_domain') >= 2:
        reasons.append(
            f'Много дефисов в домене: {int(value("num_hyphens_domain"))}'
        )
    if value('num_suspicious_words') > 0:
        reasons.append(
            'URL содержит подозрительные слова '
            f'({int(value("num_suspicious_words"))} шт.)'
        )
    if value('domain_entropy') >= 3.5:
        reasons.append('Высокая энтропия домена (похоже на случайный набор символов)')
    if value('url_length') >= 100:
        reasons.append(f'Слишком длинный URL ({int(value("url_length"))} символов)')
    if value('num_special_chars') >= 5:
        reasons.append('Много специальных символов в URL')
    if value('digit_ratio_domain') >= 0.3:
        reasons.append('Высокая доля цифр в домене')

    return reasons
