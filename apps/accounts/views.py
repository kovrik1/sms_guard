"""
Web and REST API views for user registration, authentication, and profile management
(Раздел 2.1, 4.4, 8.1, Задания 14, 15, 16).
"""
import logging
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.cache import cache
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .forms import (
    LoginForm,
    PasswordChangeCustomForm,
    ProfileSettingsForm,
    RegisterForm,
)
from .models import get_or_create_profile
from .serializers import LoginSerializer, RegisterSerializer, UserSerializer

logger = logging.getLogger("apps.accounts")


def _attempt_authentication(request: HttpRequest, username: str, password: str):
    """
    Authenticate user while enforcing ФТ-1.3 (60-second lockout after 5 consecutive failed attempts).
    Returns tuple: (user_or_none, is_locked: bool, remaining_seconds: int)
    """
    username_clean = (username or "").strip()
    cache_fail_key = f"login_fails:{username_clean.lower()}"
    cache_lock_key = f"login_lock:{username_clean.lower()}"

    max_attempts = getattr(settings, "MAX_FAILED_LOGIN_ATTEMPTS", 5)
    lockout_seconds = getattr(settings, "LOGIN_LOCKOUT_SECONDS", 60)

    user_obj = User.objects.filter(username=username_clean).first()
    if user_obj is not None:
        profile = get_or_create_profile(user_obj)
        if profile.is_locked():
            return None, True, profile.remaining_lock_seconds()
    else:
        if cache.get(cache_lock_key):
            return None, True, lockout_seconds

    user = authenticate(request, username=username_clean, password=password)
    if user is not None:
        profile = get_or_create_profile(user)
        profile.reset_failed_logins()
        cache.delete(cache_fail_key)
        cache.delete(cache_lock_key)
        return user, False, 0

    # Authentication failed
    if user_obj is not None:
        profile = get_or_create_profile(user_obj)
        profile.register_failed_login()
    else:
        fails = cache.get(cache_fail_key, 0) + 1
        cache.set(cache_fail_key, fails, timeout=lockout_seconds * 2)
        if fails >= max_attempts:
            cache.set(cache_lock_key, True, timeout=lockout_seconds)

    logger.warning("Failed login attempt for username=%s", username_clean)
    return None, False, 0


# ==============================================================================
# Web Views
# ==============================================================================


def register_view(request: HttpRequest) -> HttpResponse:
    """Web page for user registration (/register/)."""
    if request.user.is_authenticated:
        return redirect("dashboard:index")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = User.objects.create_user(
                username=form.cleaned_data["username"],
                email=form.cleaned_data["email"],
                password=form.cleaned_data["password"],
            )
            get_or_create_profile(user)
            login(request, user)
            logger.info("Registered and logged in new user id=%s", user.id)
            messages.success(request, "Регистрация прошла успешно! Добро пожаловать в SMS Guard.")
            return redirect("dashboard:index")
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {"form": form})


def login_view(request: HttpRequest) -> HttpResponse:
    """Web page for user login (/login/)."""
    if request.user.is_authenticated:
        return redirect("dashboard:index")

    status_code = 200
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            user, is_locked, remaining = _attempt_authentication(
                request, username, password
            )
            if is_locked:
                form.add_error(
                    None,
                    f"Слишком много неудачных попыток входа (5 подряд). "
                    f"Вход временно заблокирован, повторите через {remaining} с.",
                )
                status_code = 429
            elif user is not None:
                login(request, user)
                logger.info("User logged in id=%s", user.id)
                next_url = request.POST.get("next") or request.GET.get("next")
                if next_url and url_has_allowed_host_and_scheme(
                    url=next_url,
                    allowed_hosts={request.get_host()},
                    require_https=request.is_secure(),
                ):
                    return redirect(next_url)
                return redirect("dashboard:index")
            else:
                form.add_error(None, "Неверный логин или пароль.")
                status_code = 401
    else:
        form = LoginForm()

    return render(
        request,
        "accounts/login.html",
        {"form": form, "next": request.GET.get("next", "")},
        status=status_code if status_code != 401 else 200,
    )


def logout_view(request: HttpRequest) -> HttpResponse:
    """Destroy session and log out user (/logout/, ФТ-1.5)."""
    if request.user.is_authenticated:
        user_id = request.user.id
        logout(request)
        logger.info("User logged out id=%s", user_id)
    return redirect("accounts:login")


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    """User profile page (/profile/): privacy consent toggle, password change, delete history."""
    from apps.analysis.models import Check

    profile = get_or_create_profile(request.user)
    settings_form = ProfileSettingsForm(instance=profile)
    password_form = PasswordChangeCustomForm(user=request.user)

    if request.method == "POST":
        action = request.POST.get("action", "update_settings")
        if action == "update_settings":
            settings_form = ProfileSettingsForm(request.POST, instance=profile)
            if settings_form.is_valid():
                settings_form.save()
                messages.success(request, "Настройки профиля успешно сохранены.")
                return redirect("accounts:profile")
        elif action == "change_password":
            password_form = PasswordChangeCustomForm(
                user=request.user, data=request.POST
            )
            if password_form.is_valid():
                request.user.set_password(
                    password_form.cleaned_data["new_password1"]
                )
                request.user.save()
                update_session_auth_hash(request, request.user)
                messages.success(request, "Пароль успешно изменён.")
                return redirect("accounts:profile")

    total_checks = Check.objects.filter(user=request.user).count()
    return render(
        request,
        "accounts/profile.html",
        {
            "profile": profile,
            "settings_form": settings_form,
            "password_form": password_form,
            "total_checks": total_checks,
        },
    )


@login_required
@require_POST
def delete_all_history_view(request: HttpRequest) -> HttpResponse:
    """Delete all check history records belonging to the current user (Задание 16 п. 7)."""
    from apps.analysis.models import Check

    deleted_count, _ = Check.objects.filter(user=request.user).delete()
    logger.info(
        "User id=%s deleted all check history (records=%s)",
        request.user.id,
        deleted_count,
    )
    messages.success(request, "Вся ваша история проверок была успешно удалена.")
    return redirect("accounts:profile")


# ==============================================================================
# REST API Views (/api/v1/auth/...)
# ==============================================================================


class RegisterAPIView(APIView):
    """POST /api/v1/auth/register/ — User registration (ФТ-1.1, ФТ-1.2)."""

    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        get_or_create_profile(user)
        login(request, user)
        logger.info("API registration succeeded for user id=%s", user.id)
        return Response(
            UserSerializer(user).data,
            status=status.HTTP_201_CREATED,
        )


class LoginAPIView(APIView):
    """POST /api/v1/auth/login/ — User login with brute-force lockout (ФТ-1.3)."""

    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        username = serializer.validated_data["username"]
        password = serializer.validated_data["password"]

        user, is_locked, remaining = _attempt_authentication(
            request, username, password
        )
        if is_locked:
            return Response(
                {
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Слишком много неудачных попыток входа (5 подряд). "
                            f"Повторите попытку через {remaining} с."
                        ),
                    }
                },
                status=status.HTTP_429_TOO_MANY_REQUESTS,
            )

        if user is None:
            return Response(
                {
                    "error": {
                        "code": "AUTHENTICATION_ERROR",
                        "message": "Неверный логин или пароль.",
                    }
                },
                status=status.HTTP_401_UNAUTHORIZED,
            )

        login(request, user)
        logger.info("API login succeeded for user id=%s", user.id)
        return Response(
            UserSerializer(user).data,
            status=status.HTTP_200_OK,
        )


class LogoutAPIView(APIView):
    """POST /api/v1/auth/logout/ — Destroy session (ФТ-1.5)."""

    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user_id = request.user.id
        logout(request)
        logger.info("API logout for user id=%s", user_id)
        return Response(status=status.HTTP_204_NO_CONTENT)
