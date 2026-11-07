"""
Serializers for authentication and user registration REST API (ФТ-1.1, ФТ-1.2, Раздел 4.4).
"""
import re
from django.contrib.auth.models import User
from rest_framework import serializers

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")


def validate_username_rules(value: str) -> str:
    """Validate username according to ФТ-1.1 (3-30 chars, Latin letters, digits, _)."""
    value = (value or "").strip()
    if len(value) < 3 or len(value) > 30:
        raise serializers.ValidationError(
            "Логин должен содержать от 3 до 30 символов."
        )
    if not USERNAME_REGEX.match(value):
        raise serializers.ValidationError(
            "Логин может содержать только латинские буквы, цифры и символ подчёркивания (_)."
        )
    return value


def validate_password_rules(value: str) -> str:
    """Validate password according to ФТ-1.1 (at least 8 chars, letters and digits)."""
    if not value or len(value) < 8:
        raise serializers.ValidationError(
            "Пароль должен содержать не менее 8 символов."
        )
    has_letter = bool(re.search(r"[A-Za-zА-Яа-яЁё]", value))
    has_digit = bool(re.search(r"\d", value))
    if not (has_letter and has_digit):
        raise serializers.ValidationError(
            "Пароль должен содержать как буквы, так и цифры."
        )
    return value


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(
        min_length=3,
        max_length=30,
        error_messages={
            "blank": "Введите логин.",
            "required": "Поле логина обязательно.",
            "min_length": "Логин должен содержать не менее 3 символов.",
            "max_length": "Логин не должен превышать 30 символов.",
        },
    )
    email = serializers.EmailField(
        error_messages={
            "blank": "Введите адрес электронной почты.",
            "required": "Поле электронной почты обязательно.",
            "invalid": "Введите корректный адрес электронной почты.",
        },
    )
    password = serializers.CharField(
        write_only=True,
        min_length=8,
        error_messages={
            "blank": "Введите пароль.",
            "required": "Поле пароля обязательно.",
            "min_length": "Пароль должен содержать не менее 8 символов.",
        },
    )
    password_confirm = serializers.CharField(
        write_only=True,
        required=False,
        allow_blank=True,
    )

    def validate_username(self, value: str) -> str:
        value = validate_username_rules(value)
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError(
                "Пользователь с таким логином уже зарегистрирован."
            )
        return value

    def validate_email(self, value: str) -> str:
        value = (value or "").strip().lower()
        if not value:
            raise serializers.ValidationError("Введите адрес электронной почты.")
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError(
                "Пользователь с таким e-mail уже зарегистрирован."
            )
        return value

    def validate_password(self, value: str) -> str:
        return validate_password_rules(value)

    def validate(self, attrs: dict) -> dict:
        pwd_confirm = attrs.get("password_confirm")
        if pwd_confirm is not None and pwd_confirm != "" and pwd_confirm != attrs.get("password"):
            raise serializers.ValidationError(
                {"password_confirm": "Пароли не совпадают."}
            )
        return attrs

    def create(self, validated_data: dict) -> User:
        validated_data.pop("password_confirm", None)
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data["email"],
            password=validated_data["password"],
        )
        return user


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(
        error_messages={
            "blank": "Введите логин.",
            "required": "Введите логин.",
        }
    )
    password = serializers.CharField(
        write_only=True,
        error_messages={
            "blank": "Введите пароль.",
            "required": "Введите пароль.",
        },
    )


class UserSerializer(serializers.ModelSerializer):
    store_sms_text = serializers.BooleanField(
        source="profile.store_sms_text", read_only=True
    )
    preferred_language = serializers.CharField(
        source="profile.preferred_language", read_only=True
    )

    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "email",
            "is_staff",
            "store_sms_text",
            "preferred_language",
            "date_joined",
        )
