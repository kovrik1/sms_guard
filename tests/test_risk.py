"""Модульные тесты словаря социальной инженерии ``ml/rules.py`` и расчёта риска ``ml/risk.py``
(Задание 12 п. 4, Задание 13 п. 7, Раздел 8.3 ТЗ).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.risk import RiskResult, compute_risk, load_risk_config
from ml.rules import SE_GROUPS, SE_RULES, detect_se_rules
from ml.url_features import extract_url_features


# --------------------------------------------------------------------------- #
# 1. Тесты структуры словаря социальной инженерии (Задание 12 п. 4)
# --------------------------------------------------------------------------- #


def test_se_rules_structure_and_group_counts() -> None:
    """Словарь СИ содержит не менее 8 основных групп и >= 12 фраз в каждой."""
    core_groups = (
        "URGENCY",
        "MONEY_TRANSFER",
        "BANK_CARD",
        "ACCOUNT_BLOCK",
        "PRIZE",
        "CODE_REQUEST",
        "AUTHORITY",
        "SECRECY",
    )
    assert SE_GROUPS is SE_RULES
    for code in core_groups:
        assert code in SE_RULES, f"Отсутствует обязательная группа {code}"
        rule = SE_RULES[code]
        assert isinstance(rule.get("title"), str) and rule["title"]
        assert isinstance(rule.get("reason"), str) and rule["reason"]
        phrases = rule.get("phrases", ())
        assert len(phrases) >= 12, (
            f"В группе {code} должно быть не менее 12 выражений, найдено {len(phrases)}"
        )
    assert "LINK_REQUEST" in SE_RULES


def test_detect_se_rules_empty_and_output_schema() -> None:
    """Пустой ввод возвращает пустой список, а срабатывание — корректную схему словаря."""
    assert detect_se_rules("") == []
    assert detect_se_rules("   ") == []

    res = detect_se_rules("Срочно переведите деньги на банковский счёт!")
    codes = {item["code"] for item in res}
    assert {"URGENCY", "MONEY_TRANSFER", "BANK_CARD"}.issubset(codes)
    for item in res:
        assert set(item.keys()) == {"code", "title", "weight", "matched_phrases", "reason"}
        assert isinstance(item["matched_phrases"], list) and len(item["matched_phrases"]) >= 1
        assert isinstance(item["weight"], float)


# --------------------------------------------------------------------------- #
# 2. Тесты граничных значений уровней риска (34/35, 69/70)
# --------------------------------------------------------------------------- #


def test_boundary_34_is_low_safe() -> None:
    """Risk Score = 34 -> уровень LOW, категория SAFE."""
    # При отсутствии URL: score = 100 * (0.5 * p_sms + 0.2 * 0) / 0.7 = 100 * p_sms * 5 / 7
    # При p_sms = 0.34, p_url_max = 0.34, se_groups = [] -> 100 * (0.17 + 0.102) / 1.0 = 27.2
    # Чтобы получить ровно 34 без URL: p_sms * 5 / 7 = 0.34 => p_sms = 0.476
    res = compute_risk(p_sms=0.476, p_url_max=None, se_groups=[])
    assert isinstance(res, RiskResult)
    assert res.risk_score == 34
    assert res.risk_level == "LOW"
    assert res.category == "SAFE"
    assert len(res.explanation) >= 1
    assert "Оценка носит вероятностный характер" in res.disclaimer


def test_boundary_35_is_medium_suspicious() -> None:
    """Risk Score = 35 -> уровень MEDIUM, категория SUSPICIOUS."""
    # Без URL: p_sms = 0.49 -> 100 * (0.5 * 0.49) / 0.7 = 35.0
    res = compute_risk(p_sms=0.49, p_url_max=None, se_groups=[])
    assert res.risk_score == 35
    assert res.risk_level == "MEDIUM"
    assert res.category == "SUSPICIOUS"


def test_boundary_69_is_medium_suspicious() -> None:
    """Risk Score = 69 -> уровень MEDIUM, категория SUSPICIOUS."""
    # Без URL: p_sms = 0.966 -> 100 * (0.5 * 0.966) / 0.7 = 69.0
    res = compute_risk(p_sms=0.966, p_url_max=None, se_groups=[])
    assert res.risk_score == 69
    assert res.risk_level == "MEDIUM"
    assert res.category == "SUSPICIOUS"


def test_boundary_70_is_high_fraudulent() -> None:
    """Risk Score = 70 -> уровень HIGH, категория FRAUDULENT."""
    # Без URL: p_sms = 0.98 -> 100 * (0.5 * 0.98) / 0.7 = 70.0
    res = compute_risk(p_sms=0.98, p_url_max=None, se_groups=[])
    assert res.risk_score == 70
    assert res.risk_level == "HIGH"
    assert res.category == "FRAUDULENT"


# --------------------------------------------------------------------------- #
# 3. Тесты отсутствия URL и режима только URL
# --------------------------------------------------------------------------- #


def test_no_url_excludes_w2_from_denominator() -> None:
    """При p_url_max=None вес w2 исключается из числителя и знаменателя."""
    # p_sms = 0.7, 1 группа СИ (S_se = 1/3):
    # числитель = 0.5 * 0.7 + 0.2 * (1/3) = 0.35 + 0.06667 = 0.41667
    # знаменатель = 0.5 + 0.2 = 0.7 -> score = round(100 * 0.41667 / 0.7) = 60
    se = [{"code": "URGENCY", "title": "Срочность", "weight": 1.0, "reason": "Срочность"}]
    res = compute_risk(p_sms=0.7, p_url_max=None, se_groups=se)
    assert res.risk_score == 60
    assert res.risk_level == "MEDIUM"
    assert res.category == "SUSPICIOUS"


def test_url_only_mode_safe_and_dangerous() -> None:
    """В режиме проверки только ссылки (is_url_only=True) высокий уровень даёт DANGEROUS."""
    safe_res = compute_risk(p_sms=None, p_url_max=0.12, se_groups=[], url_flags={}, is_url_only=True)
    assert safe_res.risk_score == 12
    assert safe_res.risk_level == "LOW"
    assert safe_res.category == "SAFE"

    dang_res = compute_risk(
        p_sms=None,
        p_url_max=0.88,
        se_groups=[],
        url_flags={"has_punycode": 1, "num_suspicious_words": 2},
        is_url_only=True,
    )
    assert dang_res.risk_score >= 75
    assert dang_res.risk_level == "HIGH"
    assert dang_res.category == "DANGEROUS"


# --------------------------------------------------------------------------- #
# 4. Тесты всех правил усиления (overrides)
# --------------------------------------------------------------------------- #


def test_override_code_request_min_70() -> None:
    """Срабатывание группы CODE_REQUEST поднимает Risk Score минимум до 70."""
    se = detect_se_rules("Назовите код из SMS для отмены операции")
    res = compute_risk(p_sms=0.05, p_url_max=None, se_groups=se)
    assert res.risk_score >= 70
    assert res.risk_level == "HIGH"
    assert res.category == "FRAUDULENT"


def test_override_both_models_high_min_85() -> None:
    """При P_sms >= 0.8 и P_url_max >= url_t2 (0.50) итоговый Risk Score не ниже 85."""
    res = compute_risk(p_sms=0.82, p_url_max=0.66, se_groups=[])
    assert res.risk_score >= 85
    assert res.risk_level == "HIGH"
    assert res.category == "FRAUDULENT"


@pytest.mark.parametrize(
    "flag_key",
    ["has_ip_address", "has_punycode", "has_mixed_scripts", "brand_similarity"],
)
def test_override_ip_punycode_mixed_brand_min_60(flag_key: str) -> None:
    """Наличие IP-адреса, Punycode, смешанных алфавитов или маскировки под бренд даёт >= 60."""
    res = compute_risk(
        p_sms=0.05,
        p_url_max=0.15,
        se_groups=[],
        url_flags={flag_key: 1},
    )
    assert res.risk_score >= 60
    assert res.risk_level in ("MEDIUM", "HIGH")


def test_override_multiple_se_groups_without_url_min_72() -> None:
    """При срабатывании >= 2 сильных групп СИ без ссылки Risk Score >= 72."""
    se = detect_se_rules("Мама, у меня проблемы, срочно переведи 50 000, никому не говори")
    res = compute_risk(p_sms=0.10, p_url_max=None, se_groups=se, language="ru", lang_reliable=False)
    assert res.risk_score >= 72
    assert res.risk_level == "HIGH"
    assert res.category == "FRAUDULENT"


def test_override_se_plus_suspicious_url_min_75() -> None:
    """Сочетание группы СИ и подозрительной ссылки поднимает Risk Score минимум до 75."""
    se = detect_se_rules("Ваша посылка ожидает оплаты таможенного сбора")
    res = compute_risk(
        p_sms=0.20,
        p_url_max=0.45,
        se_groups=se,
        url_flags={"num_suspicious_words": 1},
    )
    assert res.risk_score >= 75
    assert res.risk_level == "HIGH"
    assert res.category == "FRAUDULENT"


# --------------------------------------------------------------------------- #
# 5. Тест динамического чтения risk_config.yaml без правки кода
# --------------------------------------------------------------------------- #


def test_dynamic_risk_config_reload_changes_result(tmp_path: Path) -> None:
    """Изменение risk_config.yaml немедленно меняет расчёт без изменения кода."""
    cfg_file = tmp_path / "custom_risk_config.yaml"
    base_cfg = load_risk_config()

    # Записываем стандартный конфиг
    cfg_file.write_text(yaml.safe_dump(base_cfg, allow_unicode=True), encoding="utf-8")
    res_default = compute_risk(
        p_sms=0.50,
        p_url_max=None,
        se_groups=[],
        config_path=cfg_file,
    )
    # 100 * (0.5 * 0.50) / 0.7 = 36 -> MEDIUM
    assert res_default.risk_score == 36
    assert res_default.risk_level == "MEDIUM"

    # Меняем границу low_max на 50 и переопределяем текст рекомендации LOW
    modified_cfg = dict(base_cfg)
    modified_cfg["low_max"] = 50
    modified_cfg["recommendations"] = dict(base_cfg["recommendations"], LOW="Кастомная рекомендация LOW")
    cfg_file.write_text(yaml.safe_dump(modified_cfg, allow_unicode=True), encoding="utf-8")

    res_updated = compute_risk(
        p_sms=0.50,
        p_url_max=None,
        se_groups=[],
        config_path=cfg_file,
    )
    assert res_updated.risk_score == 36
    assert res_updated.risk_level == "LOW"
    assert res_updated.category == "SAFE"
    assert res_updated.recommendation == "Кастомная рекомендация LOW"


def test_url_explanation_uses_correct_adjective_declension() -> None:
    """Объяснение использует винительный падеж: «как опасную» / «как подозрительную»."""
    suspicious_res = compute_risk(
        p_sms=None,
        p_url_max=0.40,
        se_groups=[],
        url_flags={},
        is_url_only=True,
    )
    assert any(
        "классифицировала ссылку как подозрительную" in reason
        for reason in suspicious_res.explanation
    )

    dangerous_res = compute_risk(
        p_sms=None,
        p_url_max=0.85,
        se_groups=[],
        url_flags={},
        is_url_only=True,
    )
    assert any(
        "классифицировала ссылку как опасную" in reason
        for reason in dangerous_res.explanation
    )


def test_language_unreliable_warning_and_max_6_reasons() -> None:
    """Для языков ru/kk добавляется предупреждение ФТ-2.8, а число причин не превышает 6."""
    se = detect_se_rules(
        "Служба безопасности банка: ваша карта заблокирована! Срочно переведите средства и назовите код из SMS, никому не говори."
    )
    res = compute_risk(
        p_sms=0.90,
        p_url_max=0.85,
        se_groups=se,
        url_flags={"has_ip_address": 1, "has_punycode": 1, "brand_similarity": 1},
        language="ru",
        lang_reliable=False,
    )
    assert 1 <= len(res.explanation) <= 6
    assert any("ФТ-2.8" in reason or "сниженную надёжность" in reason for reason in res.explanation)


# --------------------------------------------------------------------------- #
# 6. Проверка всех 13 демонстрационных примеров из Раздела 8.3 ТЗ
# --------------------------------------------------------------------------- #


def test_demo_messages_section_8_3() -> None:
    """Проверка detect_se_rules и compute_risk на всех 13 примерах из Раздела 8.3 ТЗ."""
    # 1. ru — Безопасное
    msg1 = "Привет! Встречаемся завтра в 18:00 у входа в библиотеку."
    se1 = detect_se_rules(msg1)
    assert se1 == []
    r1 = compute_risk(0.05, None, se1, language="ru", lang_reliable=False)
    assert r1.category == "SAFE"

    # 2. en — Безопасное
    msg2 = "Your order #4821 has been shipped and will arrive on Monday."
    se2 = detect_se_rules(msg2)
    assert se2 == []
    r2 = compute_risk(0.04, None, se2, language="en", lang_reliable=True)
    assert r2.category == "SAFE"

    # 3. kk — Безопасное (с предупреждением о языке)
    msg3 = "Сәлем! Ертең сағат 10:00-де кездесеміз бе?"
    se3 = detect_se_rules(msg3)
    assert se3 == []
    r3 = compute_risk(0.05, None, se3, language="kk", lang_reliable=False)
    assert r3.category == "SAFE"
    assert any("kk" in reason for reason in r3.explanation)

    # 4. ru — Легитимный код подтверждения (проверка отсутствия ложного срабатывания CODE_REQUEST!)
    msg4 = "Ваш код подтверждения: 4821. Никому не сообщайте этот код."
    se4 = detect_se_rules(msg4)
    codes4 = {g["code"] for g in se4}
    assert "CODE_REQUEST" not in codes4
    assert "SECRECY" not in codes4
    r4 = compute_risk(0.15, None, se4, language="ru", lang_reliable=False)
    assert r4.category in ("SAFE", "SUSPICIOUS")
    assert r4.risk_score < 70

    # 5. ru — Мошенническое (карта заблокирована + срочно + ссылка)
    msg5 = "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login"
    se5 = detect_se_rules(msg5)
    codes5 = {g["code"] for g in se5}
    assert {"URGENCY", "BANK_CARD", "ACCOUNT_BLOCK"}.issubset(codes5)
    u5 = extract_url_features("http://secure-bank-verify.example/login")
    r5 = compute_risk(0.25, 0.78, se5, u5, language="ru", lang_reliable=False)
    assert r5.category == "FRAUDULENT"
    assert r5.risk_score >= 75

    # 6. en — Мошенническое (URGENT + suspended + IP URL)
    msg6 = "URGENT: Your account will be suspended in 24 hours. Verify now at http://192.0.2.15/account-update"
    se6 = detect_se_rules(msg6)
    codes6 = {g["code"] for g in se6}
    assert {"URGENCY", "ACCOUNT_BLOCK"}.issubset(codes6)
    u6 = extract_url_features("http://192.0.2.15/account-update")
    r6 = compute_risk(0.88, 0.82, se6, u6, language="en", lang_reliable=True)
    assert r6.category == "FRAUDULENT"
    assert r6.risk_score >= 85

    # 7. ru — Мошенническое без ссылки («Мама, у меня проблемы, срочно переведи...»)
    msg7 = "Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори."
    se7 = detect_se_rules(msg7)
    codes7 = {g["code"] for g in se7}
    assert {"URGENCY", "MONEY_TRANSFER", "SECRECY"}.issubset(codes7)
    r7 = compute_risk(0.12, None, se7, language="ru", lang_reliable=False)
    assert r7.category == "FRAUDULENT"
    assert r7.risk_score >= 72

    # 8. en — Мошенническое (выигрыш + сокращатель bit.ly)
    msg8 = "Congratulations! You won a $1000 gift card. Claim your prize: bit.ly/prize-claimtest"
    se8 = detect_se_rules(msg8)
    codes8 = {g["code"] for g in se8}
    assert "PRIZE" in codes8
    u8 = extract_url_features("bit.ly/prize-claimtest")
    r8 = compute_risk(0.92, 0.70, se8, u8, language="en", lang_reliable=True)
    assert r8.category == "FRAUDULENT"
    assert r8.risk_score >= 85

    # 9. ru — Мошенническое без ссылки (CODE_REQUEST + AUTHORITY + ACCOUNT_BLOCK)
    msg9 = "Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции."
    se9 = detect_se_rules(msg9)
    codes9 = {g["code"] for g in se9}
    assert {"CODE_REQUEST", "AUTHORITY", "ACCOUNT_BLOCK"}.issubset(codes9)
    r9 = compute_risk(0.15, None, se9, language="ru", lang_reliable=False)
    assert r9.category == "FRAUDULENT"
    assert r9.risk_score >= 70

    # 10. kk — Подозрительное или мошенническое на казахском
    msg10 = "Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test"
    se10 = detect_se_rules(msg10)
    codes10 = {g["code"] for g in se10}
    assert {"URGENCY", "BANK_CARD", "ACCOUNT_BLOCK"}.issubset(codes10)
    u10 = extract_url_features("http://bank-kz-verify.test")
    r10 = compute_risk(0.10, 0.68, se10, u10, language="kk", lang_reliable=False)
    assert r10.category in ("SUSPICIOUS", "FRAUDULENT")
    assert r10.risk_score >= 70

    # 11. ru — Посылка + таможенный сбор + ссылка
    msg11 = "Ваша посылка ожидает оплаты таможенного сбора 350 ₸: http://post-delivery-pay.example"
    se11 = detect_se_rules(msg11)
    codes11 = {g["code"] for g in se11}
    assert "MONEY_TRANSFER" in codes11
    u11 = extract_url_features("http://post-delivery-pay.example")
    r11 = compute_risk(0.15, 0.62, se11, u11, language="ru", lang_reliable=False)
    assert r11.category in ("SUSPICIOUS", "FRAUDULENT")
    assert r11.risk_score >= 70

    # 12. URL — Безопасная ссылка
    url12 = "https://www.example.com/about"
    u12 = extract_url_features(url12)
    r12 = compute_risk(None, 0.08, [], u12, is_url_only=True)
    assert r12.category == "SAFE"
    assert r12.risk_level == "LOW"

    # 13. URL — Опасная ссылка (Punycode + ключевые слова)
    url13 = "http://xn--exmple-cua.test/secure/login?verify=1"
    u13 = extract_url_features(url13)
    r13 = compute_risk(None, 0.85, [], u13, is_url_only=True)
    assert r13.category == "DANGEROUS"
    assert r13.risk_level == "HIGH"
    assert r13.risk_score >= 75
