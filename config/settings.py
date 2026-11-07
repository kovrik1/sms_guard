"""
Django settings for SMS Guard project.
"""
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "django-insecure-sms-guard-dev-only-key-change-in-production-9f8e7d6c5b4a",
)

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv(
        "ALLOWED_HOSTS", "127.0.0.1,localhost,testserver,0.0.0.0,*"
    ).split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party apps
    "rest_framework",
    "drf_spectacular",
    # Project apps
    "apps.accounts",
    "apps.analysis",
    "apps.dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"


def _parse_database_url(db_url: str) -> dict:
    """Parse DATABASE_URL into Django DATABASES['default'] format."""
    if not db_url or db_url.startswith("sqlite"):
        path = db_url.replace("sqlite:///", "") if db_url else "db.sqlite3"
        db_path = Path(path)
        if not db_path.is_absolute():
            db_path = BASE_DIR / db_path
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": db_path,
        }
    parsed = urlparse(db_url)
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": parsed.path.lstrip("/"),
        "USER": parsed.username or "",
        "PASSWORD": parsed.password or "",
        "HOST": parsed.hostname or "localhost",
        "PORT": str(parsed.port or 5432),
    }


DATABASES = {
    "default": _parse_database_url(os.getenv("DATABASE_URL", "sqlite:///db.sqlite3"))
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
        "LOCATION": "sms-guard-cache",
    }
}

# Password hashing (ФТ-1.4: PBKDF2 with salt)
PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
]

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
        "OPTIONS": {"min_length": 8},
    },
]

# Internationalization
LANGUAGE_CODE = "ru-ru"
TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Almaty")
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Allow 'check' ForeignKey field name on UrlResult and DetectedFeature (Раздел 4.3)
SILENCED_SYSTEM_CHECKS = ["models.E020"]

# Authentication redirects
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login/"

# Security settings (ФТ-1.8, НФТ-5)
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False").lower() in (
    "true",
    "1",
    "yes",
)
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

# Login rate-limiting settings (ФТ-1.3)
MAX_FAILED_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 60

# Duplicate submission window in seconds (НФТ-3)
DUPLICATE_SUBMISSION_WINDOW_SECONDS = 3

# Django REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "EXCEPTION_HANDLER": "config.exceptions.custom_exception_handler",
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_THROTTLE_RATES": {
        "check_rate": "30/min",
    },
}

SPECTACULAR_SETTINGS = {
    "TITLE": "SMS Guard API",
    "DESCRIPTION": (
        "Интеллектуальная система обнаружения мошеннических SMS-сообщений "
        "и подозрительных ссылок"
    ),
    "VERSION": "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# Logging configuration (НФТ-9, НФТ-13, Задание 2 п. 6)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(levelname)s %(asctime)s [%(module)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 1048576,  # 1 MB
            "backupCount": 5,
            "formatter": "standard",
            "encoding": "utf-8",
        },
        "console": {
            "level": "WARNING",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "apps": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
        "ml": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["file"],
        "level": "INFO",
    },
}
