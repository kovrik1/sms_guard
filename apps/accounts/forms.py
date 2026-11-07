"""
Forms for registration, login, profile settings, and password change (ФТ-1.1, ФТ-1.2, НФТ-11).
"""
import re
from django import forms
from django.contrib.auth.models import User
from .models import UserProfile

USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_]{3,30}$")


class RegisterForm(forms.Form):
    username = forms.CharField(
        label="Логин",
        min_length=3,
        max_length=30,
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Например: ivan_petrov",
                "autocomplete": "username",
            }
        ),
        error_messages={
            "required": "Пожалуйста, введите логин.",
            "min_length": "Логин должен содержать не менее 3 символов.",
            "max_length": "Логин не должен превышать 30 символов.",
        },
    )
    email = forms.EmailField(
        label="Электронная почта (E-mail)",
        widget=forms.EmailInput(
            attrs={
                "class": "form-control",
                "placeholder": "user@example.com",
                "autocomplete": "email",
            }
        ),
        error_messages={
            "required": "Пожалуйста, введите e-mail.",
            "invalid": "Введите корректный адрес электронной почты.",
        },
    )
    password = forms.CharField(
        label="Пароль",
        min_length=8,
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control",
                "placeholder": "Минимум 8 символов, буквы и цифры",
                "autocomplete": "new-password",
            }
        ),
        error_messages={
            "required": "Пожалуйста, введите пароль.",
            "min_length": "Пароль должен содержать не менее 8 символов.",
        },
    )
    password_confirm = forms.CharField(
        label="Подтверждение пароля",
        min_length=8,
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control",
                "placeholder": "Повторите пароль",
                "autocomplete": "new-password",
            }
        ),
        error_messages={
            "required": "Пожалуйста, подтвердите пароль.",
            "min_length": "Пароль должен содержать не менее 8 символов.",
        },
    )

    def clean_username(self) -> str:
        username = (self.cleaned_data.get("username") or "").strip()
        if not USERNAME_REGEX.match(username):
            raise forms.ValidationError(
                "Логин должен содержать от 3 до 30 символов (только латинские буквы, цифры и _)."
            )
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError(
                "Пользователь с таким логином уже зарегистрирован."
            )
        return username

    def clean_email(self) -> str:
        email = (self.cleaned_data.get("email") or "").strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "Пользователь с таким e-mail уже зарегистрирован."
            )
        return email

    def clean_password(self) -> str:
        password = self.cleaned_data.get("password") or ""
        if len(password) < 8:
            raise forms.ValidationError(
                "Пароль должен содержать не менее 8 символов."
            )
        has_letter = bool(re.search(r"[A-Za-zА-Яа-яЁё]", password))
        has_digit = bool(re.search(r"\d", password))
        if not (has_letter and has_digit):
            raise forms.ValidationError(
                "Пароль должен содержать как буквы, так и цифры."
            )
        return password

    def clean(self) -> dict:
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")
        if password and password_confirm and password != password_confirm:
            self.add_error("password_confirm", "Введённые пароли не совпадают.")
        return cleaned_data


class LoginForm(forms.Form):
    username = forms.CharField(
        label="Логин",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "Введите логин",
                "autocomplete": "username",
            }
        ),
        error_messages={"required": "Введите логин."},
    )
    password = forms.CharField(
        label="Пароль",
        widget=forms.PasswordInput(
            attrs={
                "class": "form-control",
                "placeholder": "Введите пароль",
                "autocomplete": "current-password",
            }
        ),
        error_messages={"required": "Введите пароль."},
    )


class ProfileSettingsForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ("store_sms_text", "preferred_language")
        widgets = {
            "store_sms_text": forms.CheckboxInput(
                attrs={"class": "form-check-input"}
            ),
            "preferred_language": forms.Select(
                attrs={"class": "form-select"}
            ),
        }


class PasswordChangeCustomForm(forms.Form):
    old_password = forms.CharField(
        label="Текущий пароль",
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "autocomplete": "current-password"}
        ),
        error_messages={"required": "Введите текущий пароль."},
    )
    new_password1 = forms.CharField(
        label="Новый пароль",
        min_length=8,
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "autocomplete": "new-password"}
        ),
        error_messages={
            "required": "Введите новый пароль.",
            "min_length": "Новый пароль должен содержать не менее 8 символов.",
        },
    )
    new_password2 = forms.CharField(
        label="Подтверждение нового пароля",
        min_length=8,
        widget=forms.PasswordInput(
            attrs={"class": "form-control", "autocomplete": "new-password"}
        ),
        error_messages={
            "required": "Подтвердите новый пароль.",
            "min_length": "Новый пароль должен содержать не менее 8 символов.",
        },
    )

    def __init__(self, user: User, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user

    def clean_old_password(self) -> str:
        old_password = self.cleaned_data.get("old_password") or ""
        if not self.user.check_password(old_password):
            raise forms.ValidationError("Текущий пароль указан неверно.")
        return old_password

    def clean_new_password1(self) -> str:
        password = self.cleaned_data.get("new_password1") or ""
        if len(password) < 8:
            raise forms.ValidationError(
                "Пароль должен содержать не менее 8 символов."
            )
        has_letter = bool(re.search(r"[A-Za-zА-Яа-яЁё]", password))
        has_digit = bool(re.search(r"\d", password))
        if not (has_letter and has_digit):
            raise forms.ValidationError(
                "Пароль должен содержать как буквы, так и цифры."
            )
        return password

    def clean(self) -> dict:
        cleaned_data = super().clean()
        p1 = cleaned_data.get("new_password1")
        p2 = cleaned_data.get("new_password2")
        if p1 and p2 and p1 != p2:
            self.add_error("new_password2", "Новые пароли не совпадают.")
        return cleaned_data
