"""Модуль расчёта итогового уровня риска и генерации объяснений (Задание 13, раздел 2.5 ТЗ).

Реализует:
- Датакласс ``RiskResult`` с итоговой оценкой ``risk_score`` (0..100),
  уровнем риска (``LOW``, ``MEDIUM``, ``HIGH``), категорией
  (``SAFE``, ``SUSPICIOUS``, ``FRAUDULENT`` / ``DANGEROUS``), списком причин
  ``explanation`` (1..6 причин на русском языке, отсортированных по вкладу),
  рекомендацией и дисклеймером.
- Функцию ``compute_risk(...)``, динамически загружающую параметры из
  ``risk_config.yaml`` (изменение файла конфигурации сразу меняет результат
  без модификации кода).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ml.rules import SE_RULES

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_RISK_CONFIG_PATH: Path = PROJECT_ROOT / "risk_config.yaml"

DEFAULT_DISCLAIMER: str = "Оценка носит вероятностный характер и не гарантирует безопасность"

DEFAULT_RECOMMENDATIONS: dict[str, str] = {
    "LOW": (
        "Признаки мошенничества не обнаружены или незначительны. "
        "Соблюдайте базовую осторожность и никому не сообщайте коды подтверждения."
    ),
    "MEDIUM": (
        "Обнаружены подозрительные признаки. Не переходите по незнакомым ссылкам, "
        "не вводите данные карты и перепроверьте информацию по официальному номеру организации."
    ),
    "HIGH": (
        "Не переходите по ссылке, не сообщайте коды и данные карты. "
        "Свяжитесь с банком по номеру с обратной стороны карты."
    ),
}

_URL_FLAG_EXPLANATIONS: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (
        ("has_ip_address", "is_ip", "ip_address"),
        "В ссылке используется IP-адрес вместо доменного имени",
        0.90,
    ),
    (
        ("has_punycode", "punycode"),
        "Домен ссылки содержит Punycode (xn--) — возможна маскировка под известный сайт",
        0.88,
    ),
    (
        ("has_mixed_scripts", "mixed_scripts"),
        "В доменном имени смешаны разные алфавиты (кириллица и латиница)",
        0.87,
    ),
    (
        ("brand_similarity", "is_brand_spoof", "brand_spoof"),
        "Домен ссылки визуально похож на известный бренд или банк (тайпсквоттинг)",
        0.85,
    ),
    (
        ("has_at_symbol",),
        "Ссылка содержит символ @ для маскировки реального адреса перехода",
        0.75,
    ),
    (
        ("has_non_standard_port",),
        "В ссылке указан нестандартный сетевой порт",
        0.70,
    ),
    (
        ("is_shortener",),
        "Используется сервис сокращения ссылок, скрывающий конечный адрес",
        0.68,
    ),
    (
        ("is_suspicious_tld", "suspicious_tld"),
        "Домен находится в доменной зоне повышенного риска",
        0.65,
    ),
)


@dataclass
class RiskResult:
    """Результат расчёта уровня риска (Задание 13)."""

    risk_score: int
    risk_level: str
    category: str
    explanation: list[str]
    recommendation: str
    disclaimer: str = DEFAULT_DISCLAIMER
    features: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Преобразовать результат в словарь."""
        return asdict(self)


def _pick_param(
    raw_cfg: Mapping[str, Any],
    section: str,
    key: str,
    default: float | int,
) -> float | int:
    """Извлечь параметр из вложенной секции или верхнего уровня YAML.

    Если один из вариантов (верхний уровень или секция) был изменён относительно
    значения по умолчанию, возвращается изменённое значение.
    """
    sec_dict = raw_cfg.get(section)
    sec_val = sec_dict.get(key) if isinstance(sec_dict, Mapping) else None
    top_val = raw_cfg.get(key)

    if sec_val is not None and top_val is not None:
        if sec_val != default:
            return sec_val
        if top_val != default:
            return top_val
        return sec_val
    if sec_val is not None:
        return sec_val
    if top_val is not None:
        return top_val
    return default


def load_risk_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Загрузить конфигурацию расчёта риска из ``risk_config.yaml``.

    Файл перечитывается при каждом вызове, чтобы изменения в ``risk_config.yaml``
    сразу вступали в силу без правок кода и перезапуска процесса (Задание 13).
    """
    path = Path(config_path) if config_path is not None else DEFAULT_RISK_CONFIG_PATH
    raw_cfg: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, Mapping):
                raw_cfg = dict(loaded)

    overrides_raw = raw_cfg.get("overrides")
    if not isinstance(overrides_raw, Mapping):
        overrides_raw = {}

    recs_raw = raw_cfg.get("recommendations")
    if not isinstance(recs_raw, Mapping):
        recs_raw = {}

    return {
        "w1": float(_pick_param(raw_cfg, "weights", "w1", 0.5)),
        "w2": float(_pick_param(raw_cfg, "weights", "w2", 0.3)),
        "w3": float(_pick_param(raw_cfg, "weights", "w3", 0.2)),
        "se_max_groups": float(_pick_param(raw_cfg, "weights", "se_max_groups", 3)),
        "url_t1": float(_pick_param(raw_cfg, "url_thresholds", "url_t1", 0.38)),
        "url_t2": float(_pick_param(raw_cfg, "url_thresholds", "url_t2", 0.65)),
        "low_max": int(_pick_param(raw_cfg, "levels", "low_max", 34)),
        "medium_max": int(_pick_param(raw_cfg, "levels", "medium_max", 69)),
        "overrides": {
            "code_request_min_score": int(
                overrides_raw.get(
                    "code_request_min_score",
                    raw_cfg.get("code_request_min_score", 70),
                )
            ),
            "both_models_high_p_sms": float(
                overrides_raw.get(
                    "both_models_high_p_sms",
                    raw_cfg.get("both_models_high_p_sms", 0.8),
                )
            ),
            "both_models_high_min_score": int(
                overrides_raw.get(
                    "both_models_high_min_score",
                    raw_cfg.get("both_models_high_min_score", 85),
                )
            ),
            "ip_or_punycode_or_brand_min_score": int(
                overrides_raw.get(
                    "ip_or_punycode_or_brand_min_score",
                    raw_cfg.get("ip_or_punycode_or_brand_min_score", 60),
                )
            ),
            "multiple_se_groups_count": int(
                overrides_raw.get(
                    "multiple_se_groups_count",
                    raw_cfg.get("multiple_se_groups_count", 2),
                )
            ),
            "multiple_se_groups_min_score": int(
                overrides_raw.get(
                    "multiple_se_groups_min_score",
                    raw_cfg.get("multiple_se_groups_min_score", 72),
                )
            ),
            "se_plus_suspicious_url_min_score": int(
                overrides_raw.get(
                    "se_plus_suspicious_url_min_score",
                    raw_cfg.get("se_plus_suspicious_url_min_score", 75),
                )
            ),
        },
        "recommendations": {
            "LOW": str(recs_raw.get("LOW") or DEFAULT_RECOMMENDATIONS["LOW"]),
            "MEDIUM": str(recs_raw.get("MEDIUM") or DEFAULT_RECOMMENDATIONS["MEDIUM"]),
            "HIGH": str(recs_raw.get("HIGH") or DEFAULT_RECOMMENDATIONS["HIGH"]),
        },
        "disclaimer": str(raw_cfg.get("disclaimer") or DEFAULT_DISCLAIMER),
    }


def _normalize_se_groups(
    se_groups: list[Any] | tuple[Any, ...] | set[Any] | None,
) -> tuple[list[str], list[dict[str, Any]], list[tuple[float, str]]]:
    """Нормализовать список сработавших групп СИ (поддерживает и ``dict``, и ``str``)."""
    codes: list[str] = []
    features: list[dict[str, Any]] = []
    weighted_reasons: list[tuple[float, str]] = []

    if not se_groups:
        return codes, features, weighted_reasons

    for item in se_groups:
        if isinstance(item, str):
            code = item.strip().upper()
            meta = SE_RULES.get(code, {})
            title = str(meta.get("title", code))
            weight = float(meta.get("weight", 1.0))
            reason = str(meta.get("reason", f"Обнаружен признак социальной инженерии: {title}"))
        elif isinstance(item, Mapping):
            code = str(item.get("code") or item.get("group") or "SE_RULE").strip().upper()
            meta = SE_RULES.get(code, {})
            title = str(item.get("title") or meta.get("title") or code)
            weight = float(item.get("weight", meta.get("weight", 1.0)))
            reason = str(item.get("reason") or meta.get("reason") or title)
        else:
            code = str(getattr(item, "code", "SE_RULE")).strip().upper()
            meta = SE_RULES.get(code, {})
            title = str(getattr(item, "title", meta.get("title", code)))
            weight = float(getattr(item, "weight", meta.get("weight", 1.0)))
            reason = str(getattr(item, "reason", meta.get("reason", title)))

        if code and code not in codes:
            codes.append(code)
            features.append({"code": code, "title": title, "weight": weight})
            priority = 0.95 if code == "CODE_REQUEST" else (0.72 + 0.05 * min(weight, 2.0))
            weighted_reasons.append((priority, reason))

    return codes, features, weighted_reasons


def _has_high_risk_url_spoofing(flags: Mapping[str, Any]) -> bool:
    """Проверить наличие IP-адреса, Punycode, смешения алфавитов или маскировки под бренд."""
    spoof_keys = (
        "has_ip_address",
        "is_ip",
        "ip_address",
        "has_punycode",
        "punycode",
        "has_mixed_scripts",
        "mixed_scripts",
        "brand_similarity",
        "is_brand_spoof",
        "brand_spoof",
    )
    return any(bool(flags.get(k)) for k in spoof_keys)


def _has_any_suspicious_url_signal(flags: Mapping[str, Any]) -> bool:
    """Проверить наличие любого подозрительного структурного признака ссылки."""
    if _has_high_risk_url_spoofing(flags):
        return True
    signal_keys = (
        "is_shortener",
        "is_suspicious_tld",
        "suspicious_tld",
        "has_at_symbol",
        "has_non_standard_port",
        "has_suspicious_words",
    )
    if any(bool(flags.get(k)) for k in signal_keys):
        return True
    return float(flags.get("num_suspicious_words", 0) or 0) >= 1


def compute_risk(
    p_sms: float | None,
    p_url_max: float | None = None,
    se_groups: list[Any] | None = None,
    url_flags: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    is_url_only: bool = False,
    language: str = "en",
    lang_reliable: bool = True,
) -> RiskResult:
    """Вычислить итоговый Risk Score (0..100), уровень риска, категорию и объяснение.

    Базовая формула (Задание 13):
        RiskScore = 100 * (w1 * P_sms + w2 * P_url_max + w3 * S_se) / (w1 + w2 * [URL есть] + w3),
        где S_se = min(k / se_max_groups, 1).
    """
    cfg = load_risk_config(config_path)
    w1: float = cfg["w1"]
    w2: float = cfg["w2"]
    w3: float = cfg["w3"]
    se_max_groups: float = max(1.0, cfg["se_max_groups"])
    url_t1: float = cfg["url_t1"]
    url_t2: float = cfg["url_t2"]
    low_max: int = cfg["low_max"]
    medium_max: int = cfg["medium_max"]
    overrides: dict[str, Any] = cfg["overrides"]

    flags: dict[str, Any] = dict(url_flags) if isinstance(url_flags, Mapping) else {}
    codes, detected_features, se_weighted_reasons = _normalize_se_groups(se_groups)

    # Основные группы социальной инженерии (без чисто технического индикатора LINK_REQUEST)
    core_se_codes = [c for c in codes if c != "LINK_REQUEST"]
    k = len(core_se_codes) if core_se_codes else len(codes)
    s_se = min(float(k) / se_max_groups, 1.0)

    effective_url_only = bool(is_url_only or (p_sms is None and p_url_max is not None))
    p_url_val = max(0.0, min(1.0, float(p_url_max))) if p_url_max is not None else None

    lang_norm = (language or "en").strip().lower()
    is_unreliable_lang = (not lang_reliable) or (lang_norm in ("ru", "kk", "other"))

    if effective_url_only:
        base_url_score = 100.0 * (p_url_val if p_url_val is not None else 0.0)
        # Если в отдельной ссылке есть комбинация опасных признаков (например, Punycode + фишинговые слова)
        susp_words = int(flags.get("num_suspicious_words", 0) or 0)
        if _has_high_risk_url_spoofing(flags) and (susp_words >= 1 or bool(flags.get("is_suspicious_tld"))):
            base_url_score = max(base_url_score, float(overrides["se_plus_suspicious_url_min_score"]))
        raw_score = base_url_score
        p_sms_val = None
    else:
        p_sms_val = max(0.0, min(1.0, float(p_sms or 0.0)))
        # Для языков вне обучающей выборки (ru/kk) не позволяем англоязычной модели
        # занижать оценку, если сработали правила социальной инженерии
        if is_unreliable_lang and k > 0 and p_sms_val < s_se:
            p_sms_eff = s_se
        else:
            p_sms_eff = p_sms_val

        if p_url_val is not None:
            denom = w1 + w2 + w3
            raw_score = (
                100.0 * (w1 * p_sms_eff + w2 * p_url_val + w3 * s_se) / denom
                if denom > 0
                else 0.0
            )
        else:
            denom = w1 + w3
            raw_score = (
                100.0 * (w1 * p_sms_eff + w3 * s_se) / denom
                if denom > 0
                else 0.0
            )

    score = float(raw_score)

    # Применение правил усиления (overrides) из risk_config.yaml:
    # 1. Запрос кода из SMS (CODE_REQUEST) -> не ниже code_request_min_score (70)
    if "CODE_REQUEST" in codes:
        score = max(score, float(overrides["code_request_min_score"]))

    # 2. Обе модели уверены (P_sms >= 0.8 и P_url_max >= url_t2) -> не ниже both_models_high_min_score (85)
    if (
        p_sms_val is not None
        and p_sms_val >= float(overrides["both_models_high_p_sms"])
        and p_url_val is not None
        and p_url_val >= url_t2
    ):
        score = max(score, float(overrides["both_models_high_min_score"]))

    # 3. URL с IP-адресом, Punycode, смешением алфавитов или маскировкой под бренд -> не ниже 60
    if _has_high_risk_url_spoofing(flags):
        score = max(score, float(overrides["ip_or_punycode_or_brand_min_score"]))

    # 4. Несколько сильных групп СИ (>= 2, например «родственник в беде»: URGENCY + MONEY_TRANSFER + SECRECY)
    if len(core_se_codes) >= int(overrides["multiple_se_groups_count"]):
        score = max(score, float(overrides["multiple_se_groups_min_score"]))

    # 5. Сочетание социальной инженерии и подозрительной ссылки -> не ниже 75
    if len(core_se_codes) >= 1 and (
        (p_url_val is not None and p_url_val >= url_t1)
        or _has_any_suspicious_url_signal(flags)
    ):
        score = max(score, float(overrides["se_plus_suspicious_url_min_score"]))

    final_score = int(round(max(0.0, min(100.0, score))))

    # Определение уровня риска и категории по границам из risk_config.yaml
    if final_score <= low_max:
        risk_level = "LOW"
        category = "SAFE"
    elif final_score <= medium_max:
        risk_level = "MEDIUM"
        category = "SUSPICIOUS"
    else:
        risk_level = "HIGH"
        category = "DANGEROUS" if effective_url_only else "FRAUDULENT"

    # Формирование отсортированного по вкладу списка причин (1..6 причин на русском языке)
    candidates: list[tuple[float, str]] = []

    if p_sms_val is not None and p_sms_val > 0.5:
        candidates.append(
            (
                w1 * p_sms_val + 0.4,
                f"ML-модель текста выявила высокую вероятность спама/мошенничества (P_sms = {p_sms_val:.2f})",
            )
        )

    if p_url_val is not None and p_url_val >= url_t1:
        url_label = "опасной" if p_url_val >= url_t2 else "подозрительной"
        candidates.append(
            (
                w2 * p_url_val + 0.45,
                f"ML-модель классифицировала ссылку как {url_label} (P_url = {p_url_val:.2f})",
            )
        )

    candidates.extend(se_weighted_reasons)

    for flag_keys, flag_reason, flag_weight in _URL_FLAG_EXPLANATIONS:
        if any(bool(flags.get(fk)) for fk in flag_keys):
            candidates.append((flag_weight, flag_reason))

    susp_words_count = int(flags.get("num_suspicious_words", 0) or 0)
    if susp_words_count > 0 or bool(flags.get("has_suspicious_words")):
        candidates.append(
            (
                0.66,
                f"URL содержит подозрительные ключевые слова фишинга ({max(susp_words_count, 1)} шт.)",
            )
        )

    # Сортировка по убыванию вклада и удаление дубликатов
    candidates.sort(key=lambda item: item[0], reverse=True)
    explanation: list[str] = []
    for _weight, text_reason in candidates:
        if text_reason not in explanation:
            explanation.append(text_reason)
        if len(explanation) >= 6:
            break

    # Предупреждение о сниженной надёжности ML-оценки для языков ru/kk/other (ФТ-2.8)
    if not effective_url_only and is_unreliable_lang:
        lang_msg = (
            f"Предупреждение (ФТ-2.8): язык сообщения ({lang_norm}) не представлен в обучающем наборе "
            "SMS Spam Collection (en), поэтому ML-оценка текста имеет сниженную надёжность."
        )
        if lang_msg not in explanation:
            if len(explanation) >= 6:
                explanation[-1] = lang_msg
            else:
                explanation.append(lang_msg)

    if not explanation:
        if risk_level == "LOW":
            explanation.append(
                "Признаки мошенничества, социальной инженерии и опасных ссылок не обнаружены"
            )
        elif risk_level == "MEDIUM":
            explanation.append(
                "Обнаружены отдельные подозрительные признаки, требующие осторожности"
            )
        else:
            explanation.append(
                "Совокупность признаков указывает на высокую вероятность мошенничества"
            )

    recommendation = cfg["recommendations"].get(
        risk_level, DEFAULT_RECOMMENDATIONS[risk_level]
    )

    return RiskResult(
        risk_score=final_score,
        risk_level=risk_level,
        category=category,
        explanation=explanation[:6],
        recommendation=recommendation,
        disclaimer=cfg["disclaimer"],
        features=detected_features,
    )
