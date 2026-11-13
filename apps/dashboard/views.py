"""
Views and database-level aggregation for Dashboard (/) and Statistics (/stats/)
(Раздел 2.6, 4.4, 8.1, 9.2, Задание 17, ФТ-1.9, НФТ-2).
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.analysis.models import Check, DetectedFeature, ModelVersion


def _load_active_models_info() -> list[dict[str, Any]]:
    """
    Retrieve active ML models metadata from ModelVersion table,
    falling back to ml/artifacts/*_meta.json if table is empty (Задание 17 п. 8).
    """
    db_models = list(
        ModelVersion.objects.filter(is_active=True).order_by("model_type", "-trained_at")
    )
    if db_models:
        result = []
        seen_types = set()
        for m in db_models:
            if m.model_type in seen_types:
                continue
            seen_types.add(m.model_type)
            result.append(
                {
                    "model_type": m.model_type,
                    "model_type_display": m.get_model_type_display(),
                    "version": m.version,
                    "algorithm": m.algorithm,
                    "threshold": m.threshold,
                    "metrics": m.metrics_json,
                    "trained_at": m.trained_at.strftime("%Y-%m-%d %H:%M") if m.trained_at else "",
                }
            )
        return result

    # Fallback: read from ml/artifacts/*_meta.json
    artifacts_dir = Path(settings.BASE_DIR) / "ml" / "artifacts"
    fallback_models = []
    for mtype, filename, label in (
        ("sms", "sms_model_meta.json", "SMS-классификатор"),
        ("url", "url_model_meta.json", "URL-классификатор"),
    ):
        meta_path = artifacts_dir / filename
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fallback_models.append(
                    {
                        "model_type": mtype,
                        "model_type_display": label,
                        "version": str(meta.get("version", "1.0.0")),
                        "algorithm": str(meta.get("algorithm", meta.get("model_name", "ML Pipeline"))),
                        "threshold": float(meta.get("threshold", meta.get("t2", 0.5))),
                        "metrics": meta.get("test_metrics") or meta.get("metrics") or {},
                        "trained_at": str(meta.get("trained_at", meta.get("date", ""))),
                    }
                )
            except Exception:
                pass
    return fallback_models


def compute_dashboard_summary(user, days: int = 30) -> dict[str, Any]:
    """
    Compute dashboard statistics using database-level aggregation (annotate, Count, Avg, TruncDate).
    For staff/admin users (ФТ-1.9), aggregates across all users without exposing foreign SMS texts.
    """
    if days not in (7, 30, 90):
        days = 30

    is_admin = bool(user.is_staff)
    base_qs = Check.objects.all() if is_admin else Check.objects.filter(user=user)

    now = timezone.now()
    start_date = (now - timedelta(days=days - 1)).date()
    period_qs = base_qs.filter(created_at__date__gte=start_date)

    # Single DB query for all counters & average Risk Score
    agg = base_qs.aggregate(
        total_checks=Count("id"),
        total_sms=Count("id", filter=Q(check_type=Check.TYPE_SMS)),
        total_url=Count("id", filter=Q(check_type=Check.TYPE_URL)),
        safe_count=Count("id", filter=Q(category=Check.CATEGORY_SAFE)),
        suspicious_count=Count("id", filter=Q(category=Check.CATEGORY_SUSPICIOUS)),
        fraudulent_only_count=Count("id", filter=Q(category=Check.CATEGORY_FRAUDULENT)),
        dangerous_count=Count("id", filter=Q(category=Check.CATEGORY_DANGEROUS)),
        low_risk_count=Count("id", filter=Q(risk_level=Check.RISK_LOW)),
        medium_risk_count=Count("id", filter=Q(risk_level=Check.RISK_MEDIUM)),
        high_risk_count=Count("id", filter=Q(risk_level=Check.RISK_HIGH)),
        avg_risk_score=Avg("risk_score"),
    )

    total_checks = agg["total_checks"] or 0
    total_sms = agg["total_sms"] or 0
    total_url = agg["total_url"] or 0
    safe_count = agg["safe_count"] or 0
    suspicious_count = agg["suspicious_count"] or 0
    fraudulent_only = agg["fraudulent_only_count"] or 0
    dangerous_count = agg["dangerous_count"] or 0
    fraudulent_total = fraudulent_only + dangerous_count
    avg_risk_score = round(float(agg["avg_risk_score"] or 0.0), 1)

    # Daily dynamics aggregation in DB via TruncDate (Задание 17 п. 3, 7)
    daily_rows = (
        period_qs.annotate(day=TruncDate("created_at"))
        .values("day")
        .annotate(
            total=Count("id"),
            safe=Count("id", filter=Q(category=Check.CATEGORY_SAFE)),
            suspicious=Count("id", filter=Q(category=Check.CATEGORY_SUSPICIOUS)),
            fraudulent=Count(
                "id",
                filter=Q(
                    category__in=[
                        Check.CATEGORY_FRAUDULENT,
                        Check.CATEGORY_DANGEROUS,
                    ]
                ),
            ),
        )
        .order_by("day")
    )

    daily_map = {
        row["day"].strftime("%Y-%m-%d"): row
        for row in daily_rows
        if row.get("day") is not None
    }

    daily_series = []
    for i in range(days):
        d = start_date + timedelta(days=i)
        d_str = d.strftime("%Y-%m-%d")
        row = daily_map.get(d_str)
        daily_series.append(
            {
                "date": d_str,
                "label": d.strftime("%d.%m"),
                "total": row["total"] if row else 0,
                "safe": row["safe"] if row else 0,
                "suspicious": row["suspicious"] if row else 0,
                "fraudulent": row["fraudulent"] if row else 0,
            }
        )

    # Top-5 triggered social engineering / suspicious features via DB aggregation (Задание 17 п. 4, 7)
    features_qs = (
        DetectedFeature.objects.all()
        if is_admin
        else DetectedFeature.objects.filter(check__user=user)
    )
    top_features_rows = list(
        features_qs.values("code", "title")
        .annotate(count=Count("id"))
        .order_by("-count", "code")[:5]
    )

    # Last 10 checks: for privacy (ФТ-1.9), never expose other users' SMS text!
    own_recent_qs = Check.objects.filter(user=user).order_by("-created_at", "-id")[:10]
    recent_checks = [
        {
            "id": c.id,
            "type": c.check_type,
            "snippet": c.snippet,
            "category": c.category,
            "risk_level": c.risk_level,
            "risk_score": c.risk_score,
            "language": c.language,
            "created_at": c.created_at.strftime("%d.%m.%Y %H:%M"),
            "is_own": True,
        }
        for c in own_recent_qs
    ]

    # Language breakdown via DB aggregation for extended stats page
    lang_rows = list(
        base_qs.values("language")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    by_language = {r["language"]: r["count"] for r in lang_rows}

    active_models = _load_active_models_info()

    return {
        "period_days": days,
        "is_admin_view": is_admin,
        "total_checks": total_checks,
        "total_sms": total_sms,
        "total_url": total_url,
        "safe_count": safe_count,
        "suspicious_count": suspicious_count,
        "fraudulent_count": fraudulent_total,
        "dangerous_count": dangerous_count,
        "avg_risk_score": avg_risk_score,
        "by_category": {
            "SAFE": safe_count,
            "SUSPICIOUS": suspicious_count,
            "FRAUDULENT": fraudulent_total,
            "DANGEROUS": dangerous_count,
        },
        "by_risk_level": {
            "LOW": agg["low_risk_count"] or 0,
            "MEDIUM": agg["medium_risk_count"] or 0,
            "HIGH": agg["high_risk_count"] or 0,
        },
        "by_type": {
            "SMS": total_sms,
            "URL": total_url,
        },
        "by_language": by_language,
        "daily": daily_series,
        "timeline": daily_series,
        "top_features": top_features_rows,
        "recent_checks": recent_checks,
        "active_models": active_models,
    }


@login_required
def dashboard_view(request: HttpRequest) -> HttpResponse:
    """Main Dashboard page (/, Раздел 2.6, 9.2, Задание 17)."""
    try:
        days = int(request.GET.get("days", 30))
    except (TypeError, ValueError):
        days = 30
    if days not in (7, 30, 90):
        days = 30

    summary = compute_dashboard_summary(request.user, days=days)
    return render(
        request,
        "dashboard/index.html",
        {
            "summary": summary,
            "summary_json": json.dumps(summary, ensure_ascii=False),
            "selected_days": days,
        },
    )


@login_required
def stats_view(request: HttpRequest) -> HttpResponse:
    """Extended statistics page (/stats/, Раздел 8.1 п. 7)."""
    try:
        days = int(request.GET.get("days", 30))
    except (TypeError, ValueError):
        days = 30
    if days not in (7, 30, 90):
        days = 30

    summary = compute_dashboard_summary(request.user, days=days)
    return render(
        request,
        "dashboard/stats.html",
        {
            "summary": summary,
            "summary_json": json.dumps(summary, ensure_ascii=False),
            "selected_days": days,
        },
    )


class StatsSummaryAPIView(APIView):
    """GET /api/v1/stats/summary/ — JSON data for Dashboard charts and counters."""

    permission_classes = [IsAuthenticated]

    @extend_schema(
        responses={200: OpenApiTypes.OBJECT},
        description="Return aggregated dashboard statistics and chart data.",
    )
    def get(self, request, *args, **kwargs):
        try:
            days = int(request.query_params.get("days", 30))
        except (TypeError, ValueError):
            days = 30
        summary = compute_dashboard_summary(request.user, days=days)
        return Response(summary, status=status.HTTP_200_OK)
