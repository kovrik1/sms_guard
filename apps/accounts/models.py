"""
Models for user profiles and login security (Раздел 4.3, ФТ-1.3, Задание 16).
"""
from datetime import timedelta
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


class UserProfile(models.Model):
    """Extended profile for SMS Guard users."""

    LANGUAGE_CHOICES = [
        ("ru", "Русский"),
        ("kk", "Қазақша"),
        ("en", "English"),
    ]

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
        verbose_name="Пользователь",
    )
    store_sms_text = models.BooleanField(
        default=False,
        verbose_name="Хранить исходный текст SMS в истории",
    )
    preferred_language = models.CharField(
        max_length=10,
        choices=LANGUAGE_CHOICES,
        default="ru",
        verbose_name="Предпочитаемый язык",
    )
    failed_login_attempts = models.PositiveIntegerField(
        default=0,
        verbose_name="Неудачные попытки входа подряд",
    )
    locked_until = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Блокировка входа до",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "user_profile"
        verbose_name = "Профиль пользователя"
        verbose_name_plural = "Профили пользователей"

    def __str__(self) -> str:
        return f"Profile({self.user.username})"

    def is_locked(self) -> bool:
        """Check whether the user is currently locked out from logging in (ФТ-1.3)."""
        if self.locked_until and self.locked_until > timezone.now():
            return True
        return False

    def remaining_lock_seconds(self) -> int:
        """Return remaining lockout time in seconds."""
        if not self.is_locked():
            return 0
        delta = (self.locked_until - timezone.now()).total_seconds()
        return max(1, int(round(delta)))

    def register_failed_login(self) -> bool:
        """
        Increment failed login counter and lock account for 60s if threshold reached.
        Returns True if the account is now locked.
        """
        now = timezone.now()
        if self.locked_until and self.locked_until <= now:
            # Previous lock expired, reset counter before counting new failure
            self.failed_login_attempts = 0
            self.locked_until = None

        self.failed_login_attempts += 1
        max_attempts = getattr(settings, "MAX_FAILED_LOGIN_ATTEMPTS", 5)
        lockout_seconds = getattr(settings, "LOGIN_LOCKOUT_SECONDS", 60)

        if self.failed_login_attempts >= max_attempts:
            self.locked_until = now + timedelta(seconds=lockout_seconds)

        self.save(update_fields=["failed_login_attempts", "locked_until", "updated_at"])
        return self.is_locked()

    def reset_failed_logins(self) -> None:
        """Reset failed login counter and lock status after successful login."""
        if self.failed_login_attempts > 0 or self.locked_until is not None:
            self.failed_login_attempts = 0
            self.locked_until = None
            self.save(update_fields=["failed_login_attempts", "locked_until", "updated_at"])


def get_or_create_profile(user: User) -> UserProfile:
    """Safely retrieve or create a UserProfile for a given User."""
    profile, _ = UserProfile.objects.get_or_create(user=user)
    return profile


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance: User, created: bool, **kwargs):
    """Automatically create a UserProfile whenever a User is created."""
    if created:
        UserProfile.objects.get_or_create(user=instance)
