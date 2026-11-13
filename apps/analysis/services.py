"""
Unified SMS and URL analysis service (Задание 12, Задание 16, НФТ-1, НФТ-3, НФТ-9, НФТ-14).
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from django.conf import settings
from django.contrib.auth.models import User
from django.db import transaction
from django.utils import timezone

from apps.accounts.models import UserProfile, get_or_create_profile
from config.exceptions import ModelUnavailableError
from .models import (
    DISCLAIMER_TEXT,
    LANGUAGE_WARNINGS,
    Check,
    DetectedFeature,
    UrlResult,
)

logger = logging.getLogger("apps.analysis")

SE_GROUP_TITLES: dict[str, str] = {
    "URGENCY": "Срочность и психологическое давление",
    "MONEY_TRANSFER": "Требование перевода денег или оплаты",
    "BANK_CARD": "Упоминание банковской карты или счёта",
    "ACCOUNT_BLOCK": "Угроза блокировки карты или аккаунта",
    "PRIZE": "Обещание выигрыша, приза или бонуса",
    "CODE_REQUEST": "Запрос секретного кода из SMS или пароля",
    "AUTHORITY": "Маскировка под банк, госорган или службу безопасности",
    "SECRECY": "Требование секретности («никому не говорите»)",
    "LINK_REQUEST": "Побуждение перейти по внешней ссылке",
}

URL_FLAG_TITLES: dict[str, tuple[str, str, float]] = {
    "has_ip_address": (
        "URL_IP_ADDRESS",
        "В ссылке используется IP-адрес вместо доменного имени",
        1.5,
    ),
    "has_punycode": (
        "URL_PUNYCODE",
        "Домен использует Punycode-кодировку (xn--), возможна маскировка",
        1.5,
    ),
    "has_mixed_scripts": (
        "URL_MIXED_SCRIPTS",
        "Смешение алфавитов (кириллица и латиница) в доменном имени",
        1.5,
    ),
    "brand_similarity": (
        "URL_BRAND_IMPERSONATION",
        "Домен похож на известный бренд (возможный тайпсквоттинг)",
        1.4,
    ),
    "is_shortener": (
        "URL_SHORTENER",
        "Используется сервис сокращения ссылок, скрывающий конечный адрес",
        1.1,
    ),
    "is_suspicious_tld": (
        "URL_SUSPICIOUS_TLD",
        "Домен в зоне повышенного риска (.xyz, .top, .click, .test и др.)",
        1.0,
    ),
    "has_at_symbol": (
        "URL_AT_SYMBOL",
        "Символ @ в адресе ссылки (маскировка реального хоста)",
        1.2,
    ),
    "has_non_standard_port": (
        "URL_NON_STANDARD_PORT",
        "Указан нестандартный сетевой порт в URL",
        1.1,
    ),
}

DEFAULT_RECOMMENDATIONS: dict[str, str] = {
    "LOW": (
        "Явных признаков мошенничества не выявлено. Тем не менее сохраняйте бдительность "
        "и не сообщайте посторонним коды подтверждения и реквизиты карт."
    ),
    "MEDIUM": (
        "Сообщение содержит подозрительные признаки. Не переходите по незнакомым ссылкам, "
        "не вводите пароли и перепроверьте информацию через официальный сайт или приложение организации."
    ),
    "HIGH": (
        "Высокий риск мошенничества! Не переходите по ссылке, не переводите деньги и никому "
        "не сообщайте коды из SMS или данные банковской карты. Свяжитесь с организацией по официальному номеру."
    ),
}


@dataclass
class AnalysisResult:
    """Dataclass containing complete analysis output (Задание 12 п. 6, Раздел 4.4)."""

    type: str
    language: str
    category: str
    risk_level: str
    risk_score: int
    p_sms: float | None
    p_url_max: float | None
    urls: list[dict[str, Any]] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    detected_features: list[dict[str, Any]] = field(default_factory=list)
    url_features: dict[str, Any] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    recommendation: str = ""
    disclaimer: str = DISCLAIMER_TEXT
    language_warning: str = ""
    model_versions: dict[str, str] = field(
        default_factory=lambda: {"sms": "1.0.0", "url": "1.0.0"}
    )
    duration_ms: float = 0.0
    id: int | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["check_type"] = self.type
        data["p_url"] = self.p_url_max
        return data


def _extract_domain(url: str) -> str:
    """Extract hostname from URL without making any network calls."""
    raw = (url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlsplit(raw)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def _get_predictor():
    """
    Obtain singleton ModelPredictor instance.
    Raises ModelUnavailableError if models or predictor module are missing (НФТ-14).
    """
    try:
        from ml.predictor import ModelPredictor
    except ImportError as exc:
        logger.error("ml.predictor module could not be imported: %s", exc)
        raise ModelUnavailableError(
            "Модуль предсказания ML-моделей недоступен. Выполните обучение моделей."
        ) from exc

    try:
        predictor = ModelPredictor.get_instance()
    except Exception as exc:
        logger.error("Failed to initialize ModelPredictor: %s", exc)
        raise ModelUnavailableError(
            f"Не удалось загрузить модели машинного обучения: {exc}"
        ) from exc

    # Verify models are actually loaded
    is_loaded = True
    if hasattr(predictor, "is_loaded"):
        attr = getattr(predictor, "is_loaded")
        is_loaded = attr() if callable(attr) else bool(attr)
    elif hasattr(predictor, "models_loaded"):
        attr = getattr(predictor, "models_loaded")
        is_loaded = attr() if callable(attr) else bool(attr)
    elif hasattr(predictor, "sms_model") and hasattr(predictor, "url_model"):
        is_loaded = (
            getattr(predictor, "sms_model") is not None
            and getattr(predictor, "url_model") is not None
        )

    if not is_loaded:
        logger.error("ModelPredictor reports models are not loaded.")
        raise ModelUnavailableError(
            "Файлы обученных моделей отсутствуют или не загружены. Анализ временно недоступен."
        )

    return predictor


def get_health_status() -> dict[str, Any]:
    """Return health check payload for /api/v1/health/ without raising on missing models."""
    models_loaded = False
    model_versions = {"sms": "1.0.0", "url": "1.0.0"}
    details: dict[str, Any] = {}
    try:
        predictor = _get_predictor()
        models_loaded = True
        if hasattr(predictor, "get_model_versions") and callable(predictor.get_model_versions):
            mv = predictor.get_model_versions()
            if isinstance(mv, dict):
                model_versions.update({k: str(v) for k, v in mv.items()})
        elif hasattr(predictor, "model_versions") and isinstance(predictor.model_versions, dict):
            model_versions.update({k: str(v) for k, v in predictor.model_versions.items()})
        if hasattr(predictor, "get_status") and callable(predictor.get_status):
            st = predictor.get_status()
            if isinstance(st, dict):
                details = st
    except Exception as exc:
        models_loaded = False
        details = {"reason": str(exc)}

    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "model_versions": model_versions,
        **({"details": details} if details and not models_loaded else {}),
    }


def _parse_sms_prediction(raw_pred: Any) -> float:
    """Extract float probability P_sms in [0, 1] from predictor.predict_sms output."""
    if isinstance(raw_pred, (int, float)):
        return max(0.0, min(1.0, float(raw_pred)))
    if isinstance(raw_pred, Mapping):
        for key in ("probability", "p_sms", "prob", "score"):
            if key in raw_pred and raw_pred[key] is not None:
                return max(0.0, min(1.0, float(raw_pred[key])))
    for attr in ("probability", "p_sms", "prob", "score"):
        if hasattr(raw_pred, attr):
            val = getattr(raw_pred, attr)
            if val is not None:
                return max(0.0, min(1.0, float(val)))
    if isinstance(raw_pred, (tuple, list)) and len(raw_pred) >= 2:
        return max(0.0, min(1.0, float(raw_pred[1])))
    return 0.0


def _parse_url_prediction(raw_pred: Any, t1: float = 0.35, t2: float = 0.50) -> tuple[float, str]:
    """Extract (p_url, category) from predictor.predict_url output."""
    p_url = 0.0
    category = ""

    if isinstance(raw_pred, (int, float)):
        p_url = float(raw_pred)
    elif isinstance(raw_pred, Mapping):
        for key in ("probability", "p_url", "prob", "score"):
            if key in raw_pred and raw_pred[key] is not None:
                p_url = float(raw_pred[key])
                break
        category = str(raw_pred.get("category", "") or "").upper()
    else:
        for attr in ("probability", "p_url", "prob", "score"):
            if hasattr(raw_pred, attr) and getattr(raw_pred, attr) is not None:
                p_url = float(getattr(raw_pred, attr))
                break
        if hasattr(raw_pred, "category"):
            category = str(getattr(raw_pred, "category") or "").upper()

    p_url = max(0.0, min(1.0, p_url))
    if category not in ("SAFE", "SUSPICIOUS", "DANGEROUS"):
        if p_url >= t2:
            category = "DANGEROUS"
        elif p_url >= t1:
            category = "SUSPICIOUS"
        else:
            category = "SAFE"

    return round(p_url, 4), category


def _parse_se_rules(raw_se: Any) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Normalize output of ml.rules.detect_se_rules(text) into:
    (group_codes, detected_features_list, se_reasons_list)
    """
    codes: list[str] = []
    detected: list[dict[str, Any]] = []
    reasons: list[str] = []

    if not raw_se:
        return codes, detected, reasons

    items = raw_se
    if isinstance(raw_se, Mapping):
        if "matched_groups" in raw_se:
            items = raw_se["matched_groups"]
        elif "groups" in raw_se:
            items = raw_se["groups"]
        else:
            # Dict mapping code -> match info or bool
            items = []
            for k, v in raw_se.items():
                if v:
                    if isinstance(v, Mapping):
                        items.append({"code": str(k).upper(), **v})
                    else:
                        items.append(str(k).upper())

    if isinstance(items, (list, tuple, set, frozenset)):
        for item in items:
            if isinstance(item, str):
                code = item.upper()
                title = SE_GROUP_TITLES.get(code, code)
                weight = 1.0
                reason = f"Обнаружен признак социальной инженерии: {title.lower()}"
            elif isinstance(item, Mapping):
                if "matched" in item and not item["matched"]:
                    continue
                code = str(item.get("code") or item.get("group") or item.get("name") or "SE_RULE").upper()
                title = str(item.get("title") or item.get("description") or SE_GROUP_TITLES.get(code, code))
                weight = float(item.get("weight", 1.0))
                reason = str(item.get("reason") or item.get("explanation") or title)
            else:
                code = str(getattr(item, "code", getattr(item, "group", "SE_RULE"))).upper()
                title = str(getattr(item, "title", SE_GROUP_TITLES.get(code, code)))
                weight = float(getattr(item, "weight", 1.0))
                reason = str(getattr(item, "reason", title))

            if code not in codes:
                codes.append(code)
                detected.append({"code": code, "title": title, "weight": weight})
                if reason and reason not in reasons:
                    reasons.append(reason)

    return codes, detected, reasons


def _invoke_compute_risk(
    p_sms: float | None,
    p_url_max: float | None,
    se_groups: list[str],
    url_flags: dict[str, Any],
    url_reasons: list[str],
    se_reasons: list[str],
    is_url_only: bool = False,
    language: str = "en",
) -> tuple[int, str, str, list[str], str]:
    """
    Call ml.risk.compute_risk with flexible argument adaptation and return:
    (risk_score, risk_level, category, explanation, recommendation)
    """
    try:
        from ml.risk import compute_risk

        sig = inspect.signature(compute_risk)
        params = list(sig.parameters.keys())
        kwargs: dict[str, Any] = {}
        effective_p_sms = 0.0 if (p_sms is None and not is_url_only) else p_sms

        if "p_sms" in params:
            kwargs["p_sms"] = effective_p_sms
        if "p_url_max" in params:
            kwargs["p_url_max"] = p_url_max
        elif "p_url" in params:
            kwargs["p_url"] = p_url_max
        if "se_groups" in params:
            kwargs["se_groups"] = se_groups
        if "url_flags" in params:
            kwargs["url_flags"] = url_flags
        if "is_url_only" in params:
            kwargs["is_url_only"] = is_url_only
        if "language" in params:
            kwargs["language"] = language

        if len(kwargs) >= 2:
            risk_res = compute_risk(**kwargs)
        else:
            risk_res = compute_risk(effective_p_sms, p_url_max, se_groups, url_flags)

        if isinstance(risk_res, Mapping):
            score = int(round(float(risk_res.get("risk_score", 0))))
            level = str(risk_res.get("risk_level", "LOW")).upper()
            cat = str(risk_res.get("category", "SAFE")).upper()
            expl = list(risk_res.get("explanation") or risk_res.get("reasons") or [])
            rec = str(risk_res.get("recommendation") or "")
        else:
            score = int(round(float(getattr(risk_res, "risk_score", 0))))
            level = str(getattr(risk_res, "risk_level", "LOW")).upper()
            cat = str(getattr(risk_res, "category", "SAFE")).upper()
            expl = list(getattr(risk_res, "explanation", getattr(risk_res, "reasons", [])) or [])
            rec = str(getattr(risk_res, "recommendation", "") or "")
    except ImportError:
        # Fallback formula from TZ Задание 13 if ml.risk is not yet created
        w1, w2, w3 = 0.5, 0.3, 0.2
        s_se = min(len(se_groups) / 3.0, 1.0)
        if is_url_only and p_url_max is not None:
            raw_score = 100.0 * p_url_max
        elif p_url_max is not None:
            raw_score = 100.0 * (w1 * (p_sms or 0.0) + w2 * p_url_max + w3 * s_se) / (w1 + w2 + w3)
        else:
            raw_score = 100.0 * (w1 * (p_sms or 0.0) + w3 * s_se) / (w1 + w3)

        score = int(round(max(0.0, min(100.0, raw_score))))
        if "CODE_REQUEST" in se_groups:
            score = max(score, 70)
        if (p_sms or 0.0) >= 0.8 and (p_url_max or 0.0) >= 0.7:
            score = max(score, 85)
        if url_flags.get("has_ip_address") or url_flags.get("has_punycode") or url_flags.get("brand_similarity"):
            score = max(score, 60)

        if score >= 70:
            level = "HIGH"
            cat = "DANGEROUS" if is_url_only else "FRAUDULENT"
        elif score >= 35:
            level = "MEDIUM"
            cat = "SUSPICIOUS"
        else:
            level = "LOW"
            cat = "SAFE"

        expl = []
        rec = DEFAULT_RECOMMENDATIONS[level]

    score = max(0, min(100, score))
    if level not in ("LOW", "MEDIUM", "HIGH"):
        level = "HIGH" if score >= 70 else ("MEDIUM" if score >= 35 else "LOW")

    if is_url_only:
        if level == "HIGH":
            cat = "DANGEROUS"
        elif level == "MEDIUM":
            cat = "SUSPICIOUS"
        elif cat not in ("SAFE", "SUSPICIOUS", "DANGEROUS"):
            cat = "SAFE"
    else:
        if cat not in ("SAFE", "SUSPICIOUS", "FRAUDULENT"):
            cat = "FRAUDULENT" if level == "HIGH" else ("SUSPICIOUS" if level == "MEDIUM" else "SAFE")

    # Enrich explanation with SE and URL reasons if needed, cap at 6 reasons (Задание 13 п. 5)
    for r in se_reasons + url_reasons:
        if r and r not in expl and len(expl) < 6:
            expl.append(r)

    if not expl:
        if level == "LOW":
            expl = ["Подозрительных признаков мошенничества и опасных ссылок не обнаружено."]
        elif level == "MEDIUM":
            expl = ["Обнаружены отдельные подозрительные признаки в тексте или структуре ссылки."]
        else:
            expl = ["Совокупность признаков указывает на высокую вероятность мошенничества."]

    expl = expl[:6]
    if not rec:
        rec = DEFAULT_RECOMMENDATIONS.get(level, DEFAULT_RECOMMENDATIONS["LOW"])

    return score, level, cat, expl, rec


def _get_model_versions(predictor: Any) -> dict[str, str]:
    versions = {"sms": "1.0.0", "url": "1.0.0"}
    if hasattr(predictor, "get_model_versions") and callable(predictor.get_model_versions):
        mv = predictor.get_model_versions()
        if isinstance(mv, Mapping):
            versions["sms"] = str(mv.get("sms", versions["sms"]))
            versions["url"] = str(mv.get("url", versions["url"]))
    elif hasattr(predictor, "model_versions") and isinstance(predictor.model_versions, Mapping):
        versions["sms"] = str(predictor.model_versions.get("sms", versions["sms"]))
        versions["url"] = str(predictor.model_versions.get("url", versions["url"]))
    return versions


def analyze_sms(text: str) -> AnalysisResult:
    """
    Analyze an SMS message: language detection, URL extraction (up to 3),
    SMS ML model classification, social engineering rules, URL ML model classification,
    and unified Risk Score calculation (Задание 12).
    """
    t0 = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Текст SMS не может быть пустым.")

    predictor = _get_predictor()

    from ml.preprocessing import detect_language, extract_urls_from_text
    from ml.url_features import explain_url, extract_url_features

    language = detect_language(text)
    extracted_urls = extract_urls_from_text(text, max_urls=3)

    # 1. SMS Model Prediction
    try:
        raw_sms_pred = predictor.predict_sms(text)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        logger.error("Error in predictor.predict_sms: %s", exc)
        raise ModelUnavailableError(f"Ошибка выполнения SMS-модели: {exc}") from exc

    p_sms = round(_parse_sms_prediction(raw_sms_pred), 4)

    # 2. Social Engineering Rules
    se_codes: list[str] = []
    detected_features: list[dict[str, Any]] = []
    se_reasons: list[str] = []
    try:
        from ml.rules import detect_se_rules

        raw_se = detect_se_rules(text)
        se_codes, detected_features, se_reasons = _parse_se_rules(raw_se)
    except ImportError:
        logger.warning("ml.rules not found; skipping SE dictionary rules.")

    if extracted_urls and "LINK_REQUEST" not in se_codes:
        se_codes.append("LINK_REQUEST")
        detected_features.append(
            {
                "code": "LINK_REQUEST",
                "title": SE_GROUP_TITLES["LINK_REQUEST"],
                "weight": 1.0,
            }
        )

    # 3. URL Analysis for up to 3 extracted URLs
    urls_results: list[dict[str, Any]] = []
    aggregated_url_flags: dict[str, Any] = {}
    all_url_reasons: list[str] = []

    for u in extracted_urls:
        u_feats = extract_url_features(u)
        u_reasons = explain_url(u_feats)
        try:
            raw_url_pred = predictor.predict_url(u)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.error("Error in predictor.predict_url: %s", exc)
            raise ModelUnavailableError(f"Ошибка выполнения URL-модели: {exc}") from exc

        p_url, u_cat = _parse_url_prediction(raw_url_pred)
        domain = _extract_domain(u)

        urls_results.append(
            {
                "url": u,
                "domain": domain,
                "p_url": p_url,
                "category": u_cat,
                "features": u_feats,
                "reasons": u_reasons,
            }
        )
        for k, v in u_feats.items():
            if isinstance(v, (int, float)):
                aggregated_url_flags[k] = max(aggregated_url_flags.get(k, 0), v)
        for r in u_reasons:
            if r not in all_url_reasons:
                all_url_reasons.append(r)

        # Add high-signal URL flags to detected_features
        for flag_key, (f_code, f_title, f_weight) in URL_FLAG_TITLES.items():
            if u_feats.get(flag_key, 0) and f_code not in se_codes:
                se_codes.append(f_code)
                detected_features.append(
                    {"code": f_code, "title": f_title, "weight": f_weight}
                )

    p_url_max = (
        round(max(item["p_url"] for item in urls_results), 4)
        if urls_results
        else None
    )

    # 4. Risk Calculation
    pure_se_groups = [c for c in se_codes if c in SE_GROUP_TITLES and c != "LINK_REQUEST"]
    risk_score, risk_level, category, explanation, recommendation = _invoke_compute_risk(
        p_sms=p_sms,
        p_url_max=p_url_max,
        se_groups=pure_se_groups,
        url_flags=aggregated_url_flags,
        url_reasons=all_url_reasons,
        se_reasons=se_reasons,
        is_url_only=False,
        language=language,
    )

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    model_versions = _get_model_versions(predictor)
    language_warning = LANGUAGE_WARNINGS.get(language, "")

    return AnalysisResult(
        type=Check.TYPE_SMS,
        language=language,
        category=category,
        risk_level=risk_level,
        risk_score=risk_score,
        p_sms=p_sms,
        p_url_max=p_url_max,
        urls=urls_results,
        features=se_codes,
        detected_features=detected_features,
        url_features=urls_results[0]["features"] if urls_results else {},
        explanation=explanation,
        recommendation=recommendation,
        disclaimer=DISCLAIMER_TEXT,
        language_warning=language_warning,
        model_versions=model_versions,
        duration_ms=duration_ms,
    )


def analyze_url(url: str) -> AnalysisResult:
    """
    Analyze a standalone URL without opening any network connection (Задание 9, 12).
    """
    t0 = time.perf_counter()
    from ml.url_features import explain_url, extract_url_features, validate_url

    is_valid, _norm_url, error_msg = validate_url(url)
    if not is_valid:
        raise ValueError(error_msg or "Некорректный формат URL.")

    predictor = _get_predictor()

    u_feats = extract_url_features(url)
    u_reasons = explain_url(u_feats)

    try:
        raw_url_pred = predictor.predict_url(url)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        logger.error("Error in predictor.predict_url: %s", exc)
        raise ModelUnavailableError(f"Ошибка выполнения URL-модели: {exc}") from exc

    p_url, u_cat = _parse_url_prediction(raw_url_pred)
    domain = _extract_domain(url)

    feature_codes: list[str] = []
    detected_features: list[dict[str, Any]] = []
    for flag_key, (f_code, f_title, f_weight) in URL_FLAG_TITLES.items():
        if u_feats.get(flag_key, 0):
            feature_codes.append(f_code)
            detected_features.append(
                {"code": f_code, "title": f_title, "weight": f_weight}
            )
    if u_feats.get("num_suspicious_words", 0) > 0:
        feature_codes.append("URL_SUSPICIOUS_WORDS")
        detected_features.append(
            {
                "code": "URL_SUSPICIOUS_WORDS",
                "title": f"Подозрительные ключевые слова в адресе ({int(u_feats['num_suspicious_words'])} шт.)",
                "weight": 1.2,
            }
        )

    risk_score, risk_level, category, explanation, recommendation = _invoke_compute_risk(
        p_sms=None,
        p_url_max=p_url,
        se_groups=[],
        url_flags=u_feats,
        url_reasons=u_reasons,
        se_reasons=[],
        is_url_only=True,
    )

    # Align URL category and risk_score for standalone URL checks
    if u_cat == "DANGEROUS" or risk_level == "HIGH":
        category = "DANGEROUS"
        risk_level = "HIGH"
        risk_score = max(risk_score, 70)
        # A HIGH risk must never be returned with a LOW/MEDIUM explanation stub.
        low_medium_stubs = {
            "Признаки мошенничества, социальной инженерии и опасных ссылок не обнаружены",
            "Обнаружены отдельные подозрительные признаки, требующие осторожности",
            "Подозрительных признаков мошенничества и опасных ссылок не обнаружено.",
            "Обнаружены отдельные подозрительные признаки в тексте или структуре ссылки.",
        }
        meaningful = [reason for reason in explanation if reason not in low_medium_stubs]
        explanation = meaningful or [
            f"ML-модель классифицировала ссылку как опасную (P_url = {p_url:.2f})"
        ]
    elif u_cat == "SUSPICIOUS" or risk_level == "MEDIUM":
        category = "SUSPICIOUS"
        risk_level = "MEDIUM"
        risk_score = max(35, min(69, risk_score if risk_score >= 35 else int(round(p_url * 100))))
    else:
        category = "SAFE"

    # Keep recommendation consistent with the final (possibly overridden) risk level.
    try:
        from ml.risk import load_risk_config

        configured_recommendations = load_risk_config().get("recommendations", {})
    except Exception:
        configured_recommendations = {}
    recommendation = (
        configured_recommendations.get(risk_level)
        or DEFAULT_RECOMMENDATIONS.get(risk_level, DEFAULT_RECOMMENDATIONS["LOW"])
    )

    urls_results = [
        {
            "url": url.strip(),
            "domain": domain,
            "p_url": p_url,
            "category": category,
            "features": u_feats,
            "reasons": u_reasons,
        }
    ]

    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    model_versions = _get_model_versions(predictor)

    return AnalysisResult(
        type=Check.TYPE_URL,
        language="en",
        category=category,
        risk_level=risk_level,
        risk_score=risk_score,
        p_sms=None,
        p_url_max=p_url,
        urls=urls_results,
        features=feature_codes,
        detected_features=detected_features,
        url_features=u_feats,
        explanation=explanation,
        recommendation=recommendation,
        disclaimer=DISCLAIMER_TEXT,
        language_warning="",
        model_versions=model_versions,
        duration_ms=duration_ms,
    )


def save_analysis_result(
    user: User,
    result: AnalysisResult,
    raw_input: str,
    created_at_override=None,
) -> Check:
    """
    Persist AnalysisResult to database inside transaction.atomic() (Задание 16 п. 3).
    - Respects user.profile.store_sms_text (Задание 16 п. 4).
    - Prevents duplicate submissions within 3 seconds (НФТ-3).
    - Logs only check_id, category, duration_ms — NEVER logs SMS text (НФТ-9).
    """
    from ml.preprocessing import mask_sensitive_text

    normalized_raw = (raw_input or "").strip()
    submission_hash = hashlib.sha256(
        f"{user.id}:{result.type}:{normalized_raw}".encode("utf-8")
    ).hexdigest()

    window_seconds = getattr(settings, "DUPLICATE_SUBMISSION_WINDOW_SECONDS", 3)
    cutoff = timezone.now() - timedelta(seconds=window_seconds)

    with transaction.atomic():
        # Ensure the user's profile row exists and lock it to serialize
        # duplicate-submission checks on simultaneous double-click requests (НФТ-3).
        get_or_create_profile(user)
        profile = UserProfile.objects.select_for_update().get(user=user)

        if created_at_override is None:
            existing = (
                Check.objects.filter(
                    user=user,
                    submission_hash=submission_hash,
                    created_at__gte=cutoff,
                )
                .prefetch_related("url_results", "detected_features")
                .first()
            )
            if existing is not None:
                logger.info(
                    "Duplicate submission prevented: check_id=%s type=%s category=%s",
                    existing.id,
                    existing.check_type,
                    existing.category,
                )
                return existing

        if result.type == Check.TYPE_SMS:
            input_masked = mask_sensitive_text(normalized_raw)
            input_text = normalized_raw if profile.store_sms_text else None
        else:
            input_masked = normalized_raw
            input_text = normalized_raw if profile.store_sms_text else None

        check = Check.objects.create(
            user=user,
            check_type=result.type,
            input_text=input_text,
            input_masked=input_masked,
            language=result.language,
            category=result.category,
            risk_level=result.risk_level,
            risk_score=result.risk_score,
            p_sms=result.p_sms,
            p_url_max=result.p_url_max,
            explanation=result.explanation,
            recommendation=result.recommendation,
            sms_model_version=result.model_versions.get("sms", "1.0.0"),
            url_model_version=result.model_versions.get("url", "1.0.0"),
            submission_hash=submission_hash,
        )

        if created_at_override is not None:
            Check.objects.filter(pk=check.pk).update(created_at=created_at_override)
            check.created_at = created_at_override

        if result.urls:
            UrlResult.objects.bulk_create(
                [
                    UrlResult(
                        check=check,
                        url=u["url"],
                        domain=u.get("domain", "")[:255],
                        p_url=u["p_url"],
                        category=u["category"],
                        features_json=u.get("features", {}),
                    )
                    for u in result.urls
                ]
            )

        if result.detected_features:
            DetectedFeature.objects.bulk_create(
                [
                    DetectedFeature(
                        check=check,
                        code=str(df["code"])[:64],
                        title=str(df["title"])[:255],
                        weight=float(df.get("weight", 1.0)),
                    )
                    for df in result.detected_features
                ]
            )

    # Privacy-preserving log entry (НФТ-9: NEVER log SMS text!)
    logger.info(
        "Check completed: check_id=%s type=%s category=%s risk_score=%s duration_ms=%.2f",
        check.id,
        check.check_type,
        check.category,
        check.risk_score,
        result.duration_ms,
    )
    return check
