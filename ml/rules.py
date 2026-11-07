"""Словарь правил социальной инженерии для анализа SMS-сообщений (Задание 12, ФТ-2.5).

Модуль содержит:
- ``SE_RULES`` / ``SE_GROUPS`` — словарь групп социальной инженерии (8 основных
  групп + дополнительный индикатор ``LINK_REQUEST``), содержащий не менее 12
  выражений/паттернов в каждой основной группе на русском, казахском и английском
  языках.
- ``detect_se_rules(text)`` — функция поиска сработавших групп социальной
  инженерии с защитой от ложных срабатываний на легитимных банковских
  предупреждениях вида «Никому не сообщайте этот код».
"""

from __future__ import annotations

import re
from typing import Any

# Регулярное выражение для обнаружения URL / ссылок в тексте
_URL_IN_TEXT_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|clck\.ru|is\.gd|cutt\.ly)/\S+|<url>)",
    re.IGNORECASE,
)

# Паттерны антифрод-предупреждений банка («Никому не сообщайте этот код»)
_SAFE_CODE_WARNING_PATTERNS: tuple[str, ...] = (
    "никому не сообщайте",
    "никому не говорите этот код",
    "не сообщайте этот код",
    "не сообщайте код",
    "не передавайте код",
    "ешкімге айтпаңыз",
    "кодты ешкімге айтпаңыз",
    "қызметкерлеріне де айтпаңыз",
    "үшінші тұлғаларға бермеңіз",
    "ешкімге хабарламаңыз",
    "do not share this code",
    "never share this code",
    "don't share this code",
    "do not disclose this code",
)

# Явные мошеннические требования назвать/отправить/ввести код
_EXPLICIT_CODE_DEMAND_PATTERNS: tuple[str, ...] = (
    "сообщите код",
    "назовите код",
    "назовите его оператору",
    "продиктуйте код",
    "цифры кода",
    "кодом из sms",
    "кодом из смс",
    "пришлите код",
    "отправьте код",
    "скажите код",
    "введите код по ссылке",
    "введите код для отмены",
    "код для отмены",
    "код из sms",
    "код из смс",
    "share the code",
    "send the otp",
    "send the code",
    "tell us the code",
    "provide the code",
    "verification code to cancel",
    "enter the code at",
    "кодты айтыңыз",
    "смс кодты жіберіңіз",
    "растау кодын хабарлаңыз",
    "растау кодын",
    "жауап хабарламада жіберіңіз",
    "кодты хабарлаңыз",
    "кодты жіберіңіз",
)

SE_RULES: dict[str, dict[str, Any]] = {
    "URGENCY": {
        "code": "URGENCY",
        "title": "Срочность и давление по времени",
        "weight": 1.0,
        "reason": "Сообщение требует срочного действия и оказывает давление по времени",
        "phrases": (
            # Русский
            "срочно",
            "немедленно",
            "в течение 24 часов",
            "прямо сейчас",
            "до конца дня",
            "незамедлительно",
            "в ближайший час",
            "не откладывая",
            # Английский
            "urgent",
            "immediately",
            "within 24 hours",
            "in 24 hours",
            "act now",
            "expire today",
            "expires today",
            "verify now",
            "right now",
            "asap",
            "last chance",
            "final notice",
            # Казахский
            "тез арада",
            "жедел",
            "жедел түрде",
            "шұғыл",
            "дереу",
            "24 сағат ішінде",
        ),
    },
    "MONEY_TRANSFER": {
        "code": "MONEY_TRANSFER",
        "title": "Требование перевода или оплаты",
        "weight": 1.0,
        "reason": "Присутствует требование перевести деньги или оплатить сбор/комиссию",
        "phrases": (
            # Русский
            "переведите",
            "переведи",
            "перевод",
            "оплатите",
            "оплата сбора",
            "оплаты таможенного сбора",
            "таможенный сбор",
            "таможенного сбора",
            "комиссия",
            "оплаты комиссии",
            "задолженность",
            "штраф",
            "пополните счет",
            "пополните счёт",
            # Английский
            "send money",
            "transfer",
            "payment required",
            "fee",
            "wire transfer",
            "customs fee",
            "delivery fee",
            "pay now",
            # Казахский
            "аударыңыз",
            "аудар",
            "аударып жібер",
            "ақша керек",
            "депозит ашыңыз",
            "төлеңіз",
            "төлем",
            "айыппұл",
            "комиссия төлеу",
            "ақша аудар",
        ),
    },
    "BANK_CARD": {
        "code": "BANK_CARD",
        "title": "Банковская карта или счёт",
        "weight": 1.0,
        "reason": "Упоминаются банковская карта, счёт или платёжные реквизиты",
        "phrases": (
            # Русский
            "карта",
            "ваша карта",
            "банковский счёт",
            "банковский счет",
            "реквизиты",
            "cvv",
            "cvc",
            "срок действия карты",
            "пин-код",
            "банк",
            "банка",
            "подтвердите данные",
            # Английский
            "bank account",
            "credit card",
            "debit card",
            "card details",
            "bank card",
            "online banking",
            # Казахский
            "картаңыз",
            "банк шоты",
            "деректерді растаңыз",
            "сілтеме арқылы растаңыз",
            "банк картасы",
        ),
    },
    "ACCOUNT_BLOCK": {
        "code": "ACCOUNT_BLOCK",
        "title": "Угроза блокировки аккаунта или карты",
        "weight": 1.0,
        "reason": "Упоминается блокировка банковского аккаунта/карты или подозрительная операция",
        "phrases": (
            # Русский
            "заблокирована",
            "заблокирован",
            "блокировка",
            "приостановлен",
            "ограничен доступ",
            "подозрительная операция",
            "подозрительной операции",
            "попытка входа",
            "несанкционированный доступ",
            # Английский
            "suspended",
            "account will be suspended",
            "blocked",
            "locked",
            "unauthorized access",
            "compromised",
            "unusual activity",
            "security alert",
            # Казахский
            "бұғатталды",
            "бұғаттау",
            "шектелді",
            "күдікті операция",
            "аккаунтыңыз бұғатталды",
        ),
    },
    "PRIZE": {
        "code": "PRIZE",
        "title": "Выигрыш, приз или компенсация",
        "weight": 1.0,
        "reason": "Обещание выигрыша, приза, подарка или неожиданной выплаты",
        "phrases": (
            # Русский
            "выигрыш",
            "выиграли",
            "победителем",
            "приз",
            "подарок",
            "лотерея",
            "бонус",
            "компенсация",
            "выплата",
            "миллион",
            "розыгрыш",
            # Английский
            "won",
            "winner",
            "prize",
            "gift card",
            "reward",
            "congratulations",
            "lottery",
            "claim your prize",
            "cash prize",
            # Казахский
            "ұтыс",
            "сыйлық",
            "ұтып алдыңыз",
            "жүлде",
            "лотерея ұтысы",
        ),
    },
    "CODE_REQUEST": {
        "code": "CODE_REQUEST",
        "title": "Запрос кода подтверждения или пароля",
        "weight": 1.2,
        "reason": "Запрос секретного кода из SMS или одноразового пароля",
        "phrases": (
            # Русский
            "сообщите код",
            "назовите код",
            "код из sms",
            "код из смс",
            "кодом из sms",
            "кодом из смс",
            "цифры кода",
            "назовите его оператору",
            "продиктуйте код",
            "введите код подтверждения",
            "пришлите код",
            "отправьте код",
            "скажите код",
            "код для отмены",
            # Английский
            "share the code",
            "send the otp",
            "send the code",
            "tell us the code",
            "verification code to cancel",
            "provide the otp",
            "enter the verification code",
            # Казахский
            "кодты айтыңыз",
            "смс кодты жіберіңіз",
            "растау кодын хабарлаңыз",
            "растау кодын",
            "жауап хабарламада жіберіңіз",
            "кодты хабарлаңыз",
            "кодты жіберіңіз",
        ),
    },
    "AUTHORITY": {
        "code": "AUTHORITY",
        "title": "Апелляция к авторитету или ведомству",
        "weight": 1.0,
        "reason": "Отправитель представляется банком, службой безопасности, полицией или госорганом",
        "phrases": (
            # Русский
            "служба безопасности",
            "службы безопасности",
            "сотрудник службы безопасности",
            "сотрудник банка",
            "центральный банк",
            "национальный банк",
            "полиция",
            "мвд",
            "прокуратура",
            "налоговая",
            "госуслуги",
            "egov",
            "казпочта",
            # Английский
            "security department",
            "central bank",
            "police",
            "tax office",
            "government agency",
            "fraud department",
            # Казахский
            "қауіпсіздік қызметі",
            "ұлттық банк",
            "мемлекеттік қызмет",
            "салық органы",
        ),
    },
    "SECRECY": {
        "code": "SECRECY",
        "title": "Секретность или давление через близких",
        "weight": 1.0,
        "reason": "Просьба держать разговор в тайне или легенда о родственнике в беде",
        "phrases": (
            # Русский
            "никому не говори",
            "в тайне",
            "конфиденциально",
            "потом всё объясню",
            "потом все объясню",
            "мама, у меня проблемы",
            "мама у меня проблемы",
            "попала в дтп",
            "попал в дтп",
            "попал в беду",
            "попала в беду",
            "только между нами",
            # Английский
            "keep secret",
            "keep it secret",
            "don't tell anyone",
            "do not tell anyone",
            "between us only",
            "strictly confidential",
            # Казахский
            "ешкімге айтпа",
            "құпия",
            "менде мәселе",
            "кейін түсіндіремін",
        ),
    },
    "LINK_REQUEST": {
        "code": "LINK_REQUEST",
        "title": "Требование перейти по ссылке",
        "weight": 1.0,
        "reason": "Присутствует требование перейти по ссылке",
        "phrases": (
            "по ссылке",
            "перейдите по ссылке",
            "подтвердите данные по ссылке",
            "подтвердите по ссылке",
            "сілтеме арқылы",
            "мына сілтеме",
            "сілтемеге өтіңіз",
            "verify now at",
            "click here",
            "follow the link",
            "claim your prize:",
            "claim at",
        ),
    },
}

# Алиас для обратной совместимости
SE_GROUPS = SE_RULES


def _phrase_matches(text_lower: str, phrase: str) -> bool:
    """Проверить вхождение фразы с учётом границ слов для коротких токенов."""
    p = phrase.lower().strip()
    if not p:
        return False
    if p not in text_lower:
        return False
    # Для коротких однословных токенов проверяем границы слов, чтобы 'won' не срабатывало внутри 'wonderful'
    if len(p) <= 5 and re.fullmatch(r"[\w\-]+", p, flags=re.UNICODE):
        pattern = r"(?<!\w)" + re.escape(p) + r"(?!\w)"
        return bool(re.search(pattern, text_lower, flags=re.UNICODE))
    return True


def _is_benign_code_warning(text_lower: str, has_url: bool) -> bool:
    """Определить, является ли упоминание кода легитимным предупреждением банка
    («Ваш код подтверждения: 4821. Никому не сообщайте этот код.»).
    """
    has_safe_warning = any(w in text_lower for w in _SAFE_CODE_WARNING_PATTERNS)
    if not has_safe_warning:
        return False
    has_explicit_demand = any(d in text_lower for d in _EXPLICIT_CODE_DEMAND_PATTERNS)
    if has_explicit_demand or has_url:
        return False
    return True


def detect_se_rules(text: str) -> list[dict[str, Any]]:
    """Обнаружить сработавшие группы социальной инженерии в тексте SMS.

    Возвращает список словарей:
    ``{"code": str, "title": str, "weight": float, "matched_phrases": list[str], "reason": str}``
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text_lower = text.lower().replace("ё", "е")
    has_url = bool(_URL_IN_TEXT_RE.search(text))
    benign_warning = _is_benign_code_warning(text_lower, has_url=has_url)

    results: list[dict[str, Any]] = []

    for code, rule in SE_RULES.items():
        if benign_warning and code in ("CODE_REQUEST", "SECRECY", "BANK_CARD", "AUTHORITY"):
            continue

        matched_phrases: list[str] = []
        for phrase in rule["phrases"]:
            phrase_norm = phrase.lower().replace("ё", "е")
            if code == "SECRECY" and phrase_norm == "никому не говори":
                # Исключить срабатывание «никому не говори» как префикса «никому не говорите этот код»
                if "никому не говорите этот код" in text_lower or "никому не сообщайте" in text_lower:
                    continue
            if _phrase_matches(text_lower, phrase_norm):
                matched_phrases.append(phrase)

        if code == "CODE_REQUEST" and matched_phrases:
            # Если совпало только «введите код подтверждения» без ссылки и с предупреждением — пропустить
            if benign_warning:
                matched_phrases = []

        if code == "LINK_REQUEST" and not matched_phrases and has_url:
            # Если в SMS есть ссылка и сработала хотя бы одна группа СИ или есть призыв к действию
            if any(
                k in text_lower
                for k in (
                    "подтвердите",
                    "verify",
                    "claim",
                    "растаңыз",
                    "оплат",
                    "перейд",
                    "account",
                    "login",
                )
            ):
                matched_phrases.append("<URL>")

        if matched_phrases:
            results.append(
                {
                    "code": code,
                    "title": rule["title"],
                    "weight": float(rule["weight"]),
                    "matched_phrases": matched_phrases,
                    "reason": rule["reason"],
                }
            )

    return results
