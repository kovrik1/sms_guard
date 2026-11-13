"""
Web and REST API views for SMS/URL checking and check history
(Раздел 4.4, 8.1, Задания 14, 15, 16).
"""
from __future__ import annotations

import csv
from datetime import datetime
from typing import Any
from urllib.parse import urlencode, urlsplit

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from config.exceptions import ModelUnavailableError
from .models import Check
from .serializers import (
    CheckDetailSerializer,
    CheckListSerializer,
    CheckSmsInputSerializer,
    CheckUrlInputSerializer,
)
from .services import (
    analyze_sms,
    analyze_url,
    get_health_status,
    save_analysis_result,
)
from .throttles import CheckUserRateThrottle

# Metadata for the 22 URL features table (Задание 9, Раздел 9.4)
URL_FEATURE_SPEC: list[dict[str, Any]] = [
    {"key": "url_length", "num": 1, "label": "Длина URL (символов)", "type": "int", "warn_gt": 75},
    {"key": "domain_length", "num": 2, "label": "Длина домена (символов)", "type": "int", "warn_gt": 30},
    {"key": "path_length", "num": 3, "label": "Длина пути URL", "type": "int", "warn_gt": 40},
    {"key": "num_dots", "num": 4, "label": "Число точек в URL", "type": "int", "warn_gt": 3},
    {"key": "num_subdomains", "num": 5, "label": "Число поддоменов (tldextract)", "type": "int", "warn_gt": 2},
    {"key": "num_hyphens_domain", "num": 6, "label": "Число дефисов в домене", "type": "int", "warn_gt": 1},
    {"key": "num_digits_url", "num": 7, "label": "Число цифр в URL", "type": "int", "warn_gt": 6},
    {"key": "digit_ratio_domain", "num": 8, "label": "Доля цифр в домене", "type": "float", "warn_gt": 0.15},
    {"key": "num_special_chars", "num": 9, "label": "Число спецсимволов (@ ~ % = & _ ?)", "type": "int", "warn_gt": 3},
    {"key": "has_ip_address", "num": 10, "label": "IP-адрес вместо домена (IPv4/IPv6)", "type": "flag", "warn_eq": 1},
    {"key": "is_https", "num": 11, "label": "Схема HTTPS", "type": "https_flag", "warn_eq": 0},
    {"key": "has_non_standard_port", "num": 12, "label": "Нестандартный порт (не 80/443)", "type": "flag", "warn_eq": 1},
    {"key": "has_at_symbol", "num": 13, "label": "Символ @ в URL", "type": "flag", "warn_eq": 1},
    {"key": "has_double_slash_in_path", "num": 14, "label": "Двойной // в пути URL", "type": "flag", "warn_eq": 1},
    {"key": "is_shortener", "num": 15, "label": "Сервис сокращения ссылок (bit.ly и др.)", "type": "flag", "warn_eq": 1},
    {"key": "num_suspicious_words", "num": 16, "label": "Число подозрительных ключевых слов", "type": "int", "warn_gt": 0},
    {"key": "has_punycode", "num": 17, "label": "Punycode (xn--) в домене", "type": "flag", "warn_eq": 1},
    {"key": "has_mixed_scripts", "num": 18, "label": "Смешение алфавитов в домене", "type": "flag", "warn_eq": 1},
    {"key": "brand_similarity", "num": 19, "label": "Сходство с брендом (Левенштейн ≤ 2)", "type": "flag", "warn_eq": 1},
    {"key": "domain_entropy", "num": 20, "label": "Энтропия Шеннона символов домена", "type": "float", "warn_gt": 3.8},
    {"key": "is_suspicious_tld", "num": 21, "label": "Подозрительная доменная зона (.xyz, .top, .test и др.)", "type": "flag", "warn_eq": 1},
    {"key": "num_query_params", "num": 22, "label": "Число параметров запроса", "type": "int", "warn_gt": 3},
]

SMS_EXAMPLES = {
    "safe": "Привет! Встречаемся завтра в 18:00 у входа в библиотеку.",
    "suspicious": "Ваша посылка ожидает оплаты таможенного сбора 350 ₸: http://post-delivery-pay.example",
    "fraudulent": "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login",
}


def build_url_feature_rows(features_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Build enriched list of 22 URL feature rows with triggered flag for templates."""
    if not features_dict:
        return []
    rows = []
    for spec in URL_FEATURE_SPEC:
        key = spec["key"]
        raw_val = features_dict.get(key, 0)
        triggered = False
        if "warn_eq" in spec:
            triggered = int(bool(raw_val)) == spec["warn_eq"]
        elif "warn_gt" in spec:
            try:
                triggered = float(raw_val) > float(spec["warn_gt"])
            except (TypeError, ValueError):
                triggered = False

        if spec["type"] == "flag":
            display_val = "Да (1)" if raw_val else "Нет (0)"
        elif spec["type"] == "https_flag":
            display_val = "Да (HTTPS)" if raw_val else "Нет (HTTP)"
        elif spec["type"] == "float":
            display_val = f"{float(raw_val):.3f}"
        else:
            display_val = str(int(raw_val)) if isinstance(raw_val, (int, float)) else str(raw_val)

        rows.append(
            {
                "num": spec["num"],
                "key": key,
                "label": spec["label"],
                "raw_value": raw_val,
                "display_value": display_val,
                "triggered": triggered,
            }
        )
    return rows


def split_url_for_display(url: str) -> dict[str, str]:
    """
    Split a URL into (prefix, domain, suffix) so templates can render the URL
    as non-clickable text with a visually highlighted domain (ФТ-3.10).
    """
    raw = (url or "").strip()
    if not raw:
        return {"prefix": "", "domain": "", "suffix": ""}

    has_scheme = "://" in raw
    parse_target = raw if has_scheme else f"http://{raw}"
    try:
        parsed = urlsplit(parse_target)
        host = parsed.hostname or ""
        if host and host in raw.lower():
            idx = raw.lower().find(host)
            prefix = raw[:idx]
            domain = raw[idx : idx + len(host)]
            suffix = raw[idx + len(host) :]
            return {"prefix": prefix, "domain": domain, "suffix": suffix}
    except Exception:
        pass
    return {"prefix": "", "domain": raw, "suffix": ""}


def apply_history_filters(qs: QuerySet[Check], params: Mapping) -> QuerySet[Check]:
    """Apply date_from, date_to, type, category, risk_level filters to a Check queryset."""
    date_from = (params.get("date_from") or "").strip()
    date_to = (params.get("date_to") or "").strip()
    check_type = (params.get("type") or params.get("check_type") or "").strip().upper()
    category = (params.get("category") or "").strip().upper()
    risk_level = (params.get("risk_level") or "").strip().upper()

    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d").date()
            qs = qs.filter(created_at__date__gte=dt_from)
        except ValueError:
            pass

    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d").date()
            qs = qs.filter(created_at__date__lte=dt_to)
        except ValueError:
            pass

    if check_type in ("SMS", "URL"):
        qs = qs.filter(check_type=check_type)

    if category in ("SAFE", "SUSPICIOUS", "FRAUDULENT", "DANGEROUS"):
        qs = qs.filter(category=category)

    if risk_level in ("LOW", "MEDIUM", "HIGH"):
        qs = qs.filter(risk_level=risk_level)

    return qs


# ==============================================================================
# Web Views
# ==============================================================================


@login_required
def check_sms_view(request: HttpRequest) -> HttpResponse:
    """Web page for checking SMS messages (/check/sms/, Задание 15, Раздел 9.3)."""
    health = get_health_status()
    models_loaded = health.get("models_loaded", False)
    error_message = ""
    check_obj = None
    initial_text = ""

    if request.method == "POST":
        initial_text = request.POST.get("text", "")
        serializer = CheckSmsInputSerializer(data={"text": initial_text})
        if serializer.is_valid():
            try:
                clean_text = serializer.validated_data["text"]
                result = analyze_sms(clean_text)
                check_obj = save_analysis_result(request.user, result, clean_text)
            except ModelUnavailableError as exc:
                error_message = str(exc)
                models_loaded = False
            except Exception as exc:
                error_message = str(exc)
        else:
            errs = serializer.errors.get("text") or ["Некорректные данные."]
            error_message = str(errs[0])

    return render(
        request,
        "analysis/check_sms.html",
        {
            "models_loaded": models_loaded,
            "error_message": error_message,
            "check": check_obj,
            "initial_text": initial_text,
            "examples": SMS_EXAMPLES,
        },
    )


@login_required
def check_url_view(request: HttpRequest) -> HttpResponse:
    """Web page for checking standalone URLs (/check/url/, Задание 15, Раздел 9.4)."""
    health = get_health_status()
    models_loaded = health.get("models_loaded", False)
    error_message = ""
    check_obj = None
    url_feature_rows = []
    url_parts = None
    initial_url = ""

    if request.method == "POST":
        initial_url = request.POST.get("url", "")
        serializer = CheckUrlInputSerializer(data={"url": initial_url})
        if serializer.is_valid():
            try:
                clean_url = serializer.validated_data["url"]
                result = analyze_url(clean_url)
                check_obj = save_analysis_result(request.user, result, clean_url)
                url_feature_rows = build_url_feature_rows(result.url_features)
                url_parts = split_url_for_display(clean_url)
            except ModelUnavailableError as exc:
                error_message = str(exc)
                models_loaded = False
            except Exception as exc:
                error_message = str(exc)
        else:
            errs = serializer.errors.get("url") or ["Некорректный URL."]
            error_message = str(errs[0])

    return render(
        request,
        "analysis/check_url.html",
        {
            "models_loaded": models_loaded,
            "error_message": error_message,
            "check": check_obj,
            "url_feature_rows": url_feature_rows,
            "url_parts": url_parts,
            "initial_url": initial_url,
            "feature_spec": URL_FEATURE_SPEC,
        },
    )


@login_required
def check_detail_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Detailed view of a single check (/checks/<id>/, strictly owner-only ФТ-1.7)."""
    check = get_object_or_404(
        Check.objects.prefetch_related("url_results", "detected_features"),
        pk=pk,
        user=request.user,
    )

    urls_enriched = []
    for u in check.url_results.all():
        urls_enriched.append(
            {
                "obj": u,
                "parts": split_url_for_display(u.url),
                "feature_rows": build_url_feature_rows(u.features_json),
            }
        )

    return render(
        request,
        "analysis/check_detail.html",
        {
            "check": check,
            "urls_enriched": urls_enriched,
            "primary_url_parts": (
                split_url_for_display(check.display_text)
                if check.check_type == Check.TYPE_URL
                else None
            ),
        },
    )


@login_required
@require_POST
def check_delete_view(request: HttpRequest, pk: int) -> HttpResponse:
    """Delete a user's own check record (/checks/<id>/delete/, ФТ-1.7, Задание 16 п. 6)."""
    check = get_object_or_404(Check, pk=pk, user=request.user)
    check.delete()
    messages.success(request, f"Запись проверки #{pk} удалена.")
    return redirect("analysis:history")


@login_required
def history_view(request: HttpRequest) -> HttpResponse:
    """Paginated check history with filters (/history/, Задание 16 п. 5, Раздел 9.5)."""
    qs = Check.objects.filter(user=request.user).order_by("-created_at", "-id")
    qs = apply_history_filters(qs, request.GET)

    paginator = Paginator(qs, 20)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # Preserve filters in pagination links
    filter_params = {
        k: v
        for k, v in request.GET.items()
        if k in ("date_from", "date_to", "type", "category", "risk_level") and v
    }
    filter_query_string = urlencode(filter_params)

    return render(
        request,
        "analysis/history.html",
        {
            "page_obj": page_obj,
            "filters": {
                "date_from": request.GET.get("date_from", ""),
                "date_to": request.GET.get("date_to", ""),
                "type": request.GET.get("type", ""),
                "category": request.GET.get("category", ""),
                "risk_level": request.GET.get("risk_level", ""),
            },
            "filter_query_string": filter_query_string,
            "total_filtered": paginator.count,
        },
    )


@login_required
def history_export_csv_view(request: HttpRequest) -> HttpResponse:
    """Export the authenticated user's check history to CSV (/history/export/, Раздел 2.7)."""
    qs = Check.objects.filter(user=request.user).order_by("-created_at", "-id")
    qs = apply_history_filters(qs, request.GET)

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="sms_guard_history.csv"'
    # Write UTF-8 BOM so Excel opens Cyrillic cleanly
    response.write("\ufeff")

    writer = csv.writer(response)
    writer.writerow(
        [
            "ID",
            "Дата и время",
            "Тип",
            "Язык",
            "Категория",
            "Уровень риска",
            "Risk Score",
            "P_sms",
            "P_url_max",
            "Содержимое (маскированное)",
            "Рекомендация",
            "Версия SMS-модели",
            "Версия URL-модели",
        ]
    )

    for c in qs:
        writer.writerow(
            [
                c.id,
                c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                c.check_type,
                c.language,
                c.category,
                c.risk_level,
                c.risk_score,
                f"{c.p_sms:.4f}" if c.p_sms is not None else "",
                f"{c.p_url_max:.4f}" if c.p_url_max is not None else "",
                c.input_masked,
                c.recommendation,
                c.sms_model_version,
                c.url_model_version,
            ]
        )

    return response


# ==============================================================================
# REST API Views (/api/v1/...)
# ==============================================================================


class CheckSmsAPIView(APIView):
    """POST /api/v1/check/sms/ — Analyze SMS message and store result."""

    permission_classes = [IsAuthenticated]
    throttle_classes = [CheckUserRateThrottle]

    @extend_schema(
        request=CheckSmsInputSerializer,
        responses={
            200: CheckDetailSerializer,
            400: OpenApiTypes.OBJECT,
            503: OpenApiTypes.OBJECT,
        },
        description="Analyze an SMS message and persist the check result.",
    )
    def post(self, request, *args, **kwargs):
        serializer = CheckSmsInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        text = serializer.validated_data["text"]

        result = analyze_sms(text)
        check = save_analysis_result(request.user, result, text)
        check = (
            Check.objects.prefetch_related("url_results", "detected_features")
            .get(pk=check.pk)
        )
        return Response(
            CheckDetailSerializer(check).data,
            status=status.HTTP_200_OK,
        )


class CheckUrlAPIView(APIView):
    """POST /api/v1/check/url/ — Analyze standalone URL and store result."""

    permission_classes = [IsAuthenticated]
    throttle_classes = [CheckUserRateThrottle]

    @extend_schema(
        request=CheckUrlInputSerializer,
        responses={
            200: CheckDetailSerializer,
            400: OpenApiTypes.OBJECT,
            503: OpenApiTypes.OBJECT,
        },
        description="Analyze a standalone URL and persist the check result.",
    )
    def post(self, request, *args, **kwargs):
        serializer = CheckUrlInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        url = serializer.validated_data["url"]

        result = analyze_url(url)
        check = save_analysis_result(request.user, result, url)
        check = (
            Check.objects.prefetch_related("url_results", "detected_features")
            .get(pk=check.pk)
        )
        return Response(
            CheckDetailSerializer(check).data,
            status=status.HTTP_200_OK,
        )


class CheckListAPIView(generics.ListAPIView):
    """GET /api/v1/checks/ — Paginated history of current user's checks with filters."""

    permission_classes = [IsAuthenticated]
    serializer_class = CheckListSerializer

    def get_queryset(self) -> QuerySet[Check]:
        qs = (
            Check.objects.filter(user=self.request.user)
            .prefetch_related("url_results", "detected_features")
            .order_by("-created_at", "-id")
        )
        return apply_history_filters(qs, self.request.query_params)


class CheckDetailDeleteAPIView(APIView):
    """
    GET /api/v1/checks/<id>/ — Retrieve check details (owner only, 404 otherwise).
    DELETE /api/v1/checks/<id>/ — Delete check record (owner only, 404 otherwise).
    """

    permission_classes = [IsAuthenticated]

    def get_object(self, request, pk: int) -> Check:
        return get_object_or_404(
            Check.objects.prefetch_related("url_results", "detected_features"),
            pk=pk,
            user=request.user,
        )

    @extend_schema(
        responses={200: CheckDetailSerializer, 404: OpenApiTypes.OBJECT},
        description="Retrieve a check belonging to the current user.",
    )
    def get(self, request, pk: int, *args, **kwargs):
        check = self.get_object(request, pk)
        return Response(
            CheckDetailSerializer(check).data,
            status=status.HTTP_200_OK,
        )

    @extend_schema(
        request=None,
        responses={
            204: OpenApiResponse(description="Check deleted."),
            404: OpenApiTypes.OBJECT,
        },
        description="Delete a check belonging to the current user.",
    )
    def delete(self, request, pk: int, *args, **kwargs):
        check = self.get_object(request, pk)
        check.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class HealthAPIView(APIView):
    """GET /api/v1/health/ — Application and ML models status."""

    permission_classes = [AllowAny]

    @extend_schema(
        responses={200: OpenApiTypes.OBJECT},
        description="Return application and ML model availability status.",
    )
    def get(self, request, *args, **kwargs):
        return Response(get_health_status(), status=status.HTTP_200_OK)
