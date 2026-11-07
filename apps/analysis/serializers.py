"""
Serializers for SMS and URL analysis API endpoints (Раздел 4.4, Задание 14).
"""
import re
from rest_framework import serializers
from .models import Check, DetectedFeature, UrlResult

# Control characters except \t (\x09), \n (\x0a), \r (\x0d)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class CheckSmsInputSerializer(serializers.Serializer):
    """Validate incoming SMS text (ФТ-2.1, НФТ-12)."""

    text = serializers.CharField(
        required=True,
        allow_blank=True,
        trim_whitespace=False,
        error_messages={
            "required": "Поле text обязательно для заполнения.",
            "null": "Текст SMS не может быть пустым.",
        },
    )

    def validate_text(self, value: str) -> str:
        if not isinstance(value, str):
            raise serializers.ValidationError("Текст SMS должен быть строкой.")

        # Reject binary null bytes explicitly
        if "\x00" in value:
            raise serializers.ValidationError(
                "Текст содержит недопустимые бинарные символы."
            )

        # Strip non-printable control characters while keeping newlines/tabs
        cleaned = CONTROL_CHARS_RE.sub("", value)

        if not cleaned or not cleaned.strip():
            raise serializers.ValidationError("Текст SMS не может быть пустым.")

        if len(cleaned) > 1000:
            raise serializers.ValidationError(
                "Длина текста SMS не должна превышать 1000 символов."
            )

        return cleaned


class CheckUrlInputSerializer(serializers.Serializer):
    """Validate incoming URL string (ФТ-3.1, НФТ-12)."""

    url = serializers.CharField(
        required=True,
        allow_blank=True,
        trim_whitespace=True,
        error_messages={
            "required": "Поле url обязательно для заполнения.",
            "null": "URL не может быть пустым.",
        },
    )

    def validate_url(self, value: str) -> str:
        if not isinstance(value, str):
            raise serializers.ValidationError("Ссылка должна быть строкой.")

        if "\x00" in value:
            raise serializers.ValidationError(
                "URL содержит недопустимые бинарные символы."
            )

        cleaned = CONTROL_CHARS_RE.sub("", value).strip()
        if not cleaned:
            raise serializers.ValidationError("Введите URL для проверки.")

        if len(cleaned) > 2048:
            raise serializers.ValidationError(
                "Длина URL не должна превышать 2048 символов."
            )

        from ml.url_features import validate_url

        is_valid, _normalized, error_msg = validate_url(cleaned)
        if not is_valid:
            raise serializers.ValidationError(
                error_msg or "Некорректный формат интернет-ссылки (URL)."
            )

        return cleaned


class UrlResultSerializer(serializers.ModelSerializer):
    features = serializers.JSONField(source="features_json", read_only=True)

    class Meta:
        model = UrlResult
        fields = ("url", "domain", "p_url", "category", "features")


class DetectedFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectedFeature
        fields = ("code", "title", "weight")


class CheckDetailSerializer(serializers.ModelSerializer):
    """Full representation of a Check matching Section 4.4 of TZ."""

    type = serializers.CharField(source="check_type", read_only=True)
    urls = UrlResultSerializer(source="url_results", many=True, read_only=True)
    features = serializers.SerializerMethodField()
    detected_features = DetectedFeatureSerializer(many=True, read_only=True)
    url_features = serializers.SerializerMethodField()
    p_url = serializers.FloatField(source="p_url_max", read_only=True)
    disclaimer = serializers.CharField(read_only=True)
    language_warning = serializers.CharField(read_only=True)
    display_text = serializers.CharField(read_only=True)
    snippet = serializers.CharField(read_only=True)
    model_versions = serializers.SerializerMethodField()

    class Meta:
        model = Check
        fields = (
            "id",
            "type",
            "check_type",
            "language",
            "category",
            "risk_level",
            "risk_score",
            "p_sms",
            "p_url",
            "p_url_max",
            "urls",
            "features",
            "detected_features",
            "url_features",
            "explanation",
            "recommendation",
            "disclaimer",
            "language_warning",
            "input_masked",
            "input_text",
            "display_text",
            "snippet",
            "model_versions",
            "created_at",
        )

    def get_features(self, obj: Check) -> list[str]:
        return [df.code for df in obj.detected_features.all()]

    def get_url_features(self, obj: Check) -> dict:
        first_url = obj.url_results.first()
        if first_url and isinstance(first_url.features_json, dict):
            return first_url.features_json
        return {}

    def get_model_versions(self, obj: Check) -> dict[str, str]:
        return {
            "sms": obj.sms_model_version,
            "url": obj.url_model_version,
        }


class CheckListSerializer(serializers.ModelSerializer):
    """Compact representation of a Check for paginated history lists."""

    type = serializers.CharField(source="check_type", read_only=True)
    snippet = serializers.CharField(read_only=True)
    display_text = serializers.CharField(read_only=True)
    model_versions = serializers.SerializerMethodField()

    class Meta:
        model = Check
        fields = (
            "id",
            "type",
            "check_type",
            "snippet",
            "display_text",
            "input_masked",
            "language",
            "category",
            "risk_level",
            "risk_score",
            "p_sms",
            "p_url_max",
            "explanation",
            "recommendation",
            "model_versions",
            "created_at",
        )

    def get_model_versions(self, obj: Check) -> dict[str, str]:
        return {
            "sms": obj.sms_model_version,
            "url": obj.url_model_version,
        }
