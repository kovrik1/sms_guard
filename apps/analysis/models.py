"""
Database models for SMS and URL checks, extracted features, and ML model versions
(Раздел 4.3, Задание 16).
"""
from django.contrib.auth.models import User
from django.db import models

DISCLAIMER_TEXT = "Оценка носит вероятностный характер и не гарантирует безопасность"

LANGUAGE_WARNINGS = {
    "ru": (
        "Обучающий корпус UCI SMS Spam Collection содержит преимущественно англоязычные сообщения. "
        "Для русского языка оценка ML-модели текста дополняется словарём признаков социальной инженерии и анализом URL."
    ),
    "kk": (
        "Обучающий корпус не содержит казахскоязычных SMS. "
        "Для казахского языка работают словарь правил социальной инженерии и структурный анализ URL; "
        "вероятностная оценка текста имеет сниженную надёжность."
    ),
    "other": (
        "Язык сообщения не представлен в основном обучающем наборе данных; "
        "надёжность текстовой ML-оценки может быть снижена."
    ),
}


class Check(models.Model):
    """Result of an SMS or URL check performed by a user."""

    TYPE_SMS = "SMS"
    TYPE_URL = "URL"
    CHECK_TYPE_CHOICES = [
        (TYPE_SMS, "SMS"),
        (TYPE_URL, "URL"),
    ]

    CATEGORY_SAFE = "SAFE"
    CATEGORY_SUSPICIOUS = "SUSPICIOUS"
    CATEGORY_FRAUDULENT = "FRAUDULENT"
    CATEGORY_DANGEROUS = "DANGEROUS"
    CATEGORY_CHOICES = [
        (CATEGORY_SAFE, "Безопасное"),
        (CATEGORY_SUSPICIOUS, "Подозрительное"),
        (CATEGORY_FRAUDULENT, "Мошенническое"),
        (CATEGORY_DANGEROUS, "Потенциально опасная ссылка"),
    ]

    RISK_LOW = "LOW"
    RISK_MEDIUM = "MEDIUM"
    RISK_HIGH = "HIGH"
    RISK_LEVEL_CHOICES = [
        (RISK_LOW, "Низкий"),
        (RISK_MEDIUM, "Средний"),
        (RISK_HIGH, "Высокий"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="checks",
        verbose_name="Пользователь",
    )
    check_type = models.CharField(
        max_length=10,
        choices=CHECK_TYPE_CHOICES,
        default=TYPE_SMS,
        verbose_name="Тип проверки",
    )
    input_text = models.TextField(
        null=True,
        blank=True,
        verbose_name="Исходный текст (при согласии)",
    )
    input_masked = models.TextField(
        verbose_name="Маскированный текст или URL",
    )
    language = models.CharField(
        max_length=10,
        default="en",
        verbose_name="Язык",
    )
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default=CATEGORY_SAFE,
        verbose_name="Категория",
    )
    risk_level = models.CharField(
        max_length=15,
        choices=RISK_LEVEL_CHOICES,
        default=RISK_LOW,
        verbose_name="Уровень риска",
    )
    risk_score = models.IntegerField(
        default=0,
        verbose_name="Risk Score (0–100)",
    )
    p_sms = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Вероятность мошенничества SMS (P_sms)",
    )
    p_url_max = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Максимальная вероятность опасности URL (P_url_max)",
    )
    explanation = models.JSONField(
        default=list,
        verbose_name="Причины оценки",
    )
    recommendation = models.TextField(
        verbose_name="Рекомендация пользователю",
    )
    sms_model_version = models.CharField(
        max_length=32,
        default="1.0.0",
        verbose_name="Версия SMS-модели",
    )
    url_model_version = models.CharField(
        max_length=32,
        default="1.0.0",
        verbose_name="Версия URL-модели",
    )
    submission_hash = models.CharField(
        max_length=64,
        blank=True,
        default="",
        db_index=True,
        verbose_name="Хеш отправки (защита от дублей)",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Дата и время проверки",
    )

    class Meta:
        db_table = "checks"
        ordering = ["-created_at", "-id"]
        verbose_name = "Проверка"
        verbose_name_plural = "Проверки"
        indexes = [
            models.Index(
                fields=["user", "created_at"], name="idx_checks_user_created"
            ),
            models.Index(fields=["category"], name="idx_checks_category"),
        ]

    def __str__(self) -> str:
        return f"Check #{self.id} ({self.check_type} - {self.category}, score={self.risk_score})"

    @property
    def display_text(self) -> str:
        """Return stored text if user consented, otherwise masked text."""
        return self.input_text if self.input_text else self.input_masked

    @property
    def snippet(self) -> str:
        """Return up to 60 characters snippet for history table (Задание 16 п. 5)."""
        txt = self.display_text or ""
        if len(txt) <= 60:
            return txt
        return txt[:57] + "..."

    @property
    def disclaimer(self) -> str:
        return DISCLAIMER_TEXT

    @property
    def language_warning(self) -> str:
        if self.check_type == self.TYPE_SMS and self.language in LANGUAGE_WARNINGS:
            return LANGUAGE_WARNINGS[self.language]
        return ""


class UrlResult(models.Model):
    """Analysis result for an individual URL extracted from an SMS or checked standalone."""

    CATEGORY_SAFE = "SAFE"
    CATEGORY_SUSPICIOUS = "SUSPICIOUS"
    CATEGORY_DANGEROUS = "DANGEROUS"
    CATEGORY_CHOICES = [
        (CATEGORY_SAFE, "Безопасная"),
        (CATEGORY_SUSPICIOUS, "Подозрительная"),
        (CATEGORY_DANGEROUS, "Потенциально опасная"),
    ]

    check = models.ForeignKey(
        Check,
        on_delete=models.CASCADE,
        related_name="url_results",
        verbose_name="Проверка",
    )
    url = models.TextField(verbose_name="URL")
    domain = models.CharField(
        max_length=255,
        blank=True,
        default="",
        verbose_name="Домен",
    )
    p_url = models.FloatField(
        default=0.0,
        verbose_name="Вероятность опасности (P_url)",
    )
    category = models.CharField(
        max_length=20,
        choices=CATEGORY_CHOICES,
        default=CATEGORY_SAFE,
        verbose_name="Категория ссылки",
    )
    features_json = models.JSONField(
        default=dict,
        verbose_name="Признаки URL",
    )

    class Meta:
        db_table = "url_results"
        verbose_name = "Результат анализа URL"
        verbose_name_plural = "Результаты анализа URL"

    def __str__(self) -> str:
        return f"UrlResult({self.domain or self.url[:30]}, p={self.p_url:.2f})"


class DetectedFeature(models.Model):
    """Detected social engineering group or suspicious indicator for a Check."""

    check = models.ForeignKey(
        Check,
        on_delete=models.CASCADE,
        related_name="detected_features",
        verbose_name="Проверка",
    )
    code = models.CharField(
        max_length=64,
        verbose_name="Код признака",
    )
    title = models.CharField(
        max_length=255,
        verbose_name="Описание признака",
    )
    weight = models.FloatField(
        default=1.0,
        verbose_name="Вес признака",
    )

    class Meta:
        db_table = "detected_features"
        verbose_name = "Обнаруженный признак"
        verbose_name_plural = "Обнаруженные признаки"

    def __str__(self) -> str:
        return f"{self.code} ({self.title})"


class ModelVersion(models.Model):
    """Metadata and test metrics for trained ML models (Раздел 4.3, Задание 17 п. 8)."""

    MODEL_TYPE_SMS = "sms"
    MODEL_TYPE_URL = "url"
    MODEL_TYPE_CHOICES = [
        (MODEL_TYPE_SMS, "SMS-классификатор"),
        (MODEL_TYPE_URL, "URL-классификатор"),
    ]

    model_type = models.CharField(
        max_length=16,
        choices=MODEL_TYPE_CHOICES,
        verbose_name="Тип модели",
    )
    version = models.CharField(
        max_length=32,
        default="1.0.0",
        verbose_name="Версия",
    )
    algorithm = models.CharField(
        max_length=128,
        verbose_name="Алгоритм",
    )
    metrics_json = models.JSONField(
        default=dict,
        verbose_name="Метрики качества",
    )
    threshold = models.FloatField(
        default=0.5,
        verbose_name="Порог классификации",
    )
    trained_at = models.DateTimeField(
        verbose_name="Дата обучения",
    )
    is_active = models.BooleanField(
        default=True,
        verbose_name="Активная версия",
    )

    class Meta:
        db_table = "model_versions"
        ordering = ["model_type", "-trained_at"]
        verbose_name = "Версия ML-модели"
        verbose_name_plural = "Версии ML-моделей"

    def __str__(self) -> str:
        return f"{self.model_type.upper()} v{self.version} ({self.algorithm})"
