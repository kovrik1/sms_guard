from django.contrib import admin
from .models import Check, DetectedFeature, ModelVersion, UrlResult


class UrlResultInline(admin.TabularInline):
    model = UrlResult
    extra = 0


class DetectedFeatureInline(admin.TabularInline):
    model = DetectedFeature
    extra = 0


@admin.register(Check)
class CheckAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "check_type",
        "category",
        "risk_level",
        "risk_score",
        "language",
        "created_at",
    )
    list_filter = ("check_type", "category", "risk_level", "language")
    search_fields = ("user__username", "input_masked")
    inlines = [UrlResultInline, DetectedFeatureInline]


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = (
        "model_type",
        "version",
        "algorithm",
        "threshold",
        "is_active",
        "trained_at",
    )
    list_filter = ("model_type", "is_active")
