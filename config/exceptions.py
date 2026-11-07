"""
Unified exception handling for SMS Guard REST API (Раздел 4.4, НФТ-12, НФТ-14).
"""
import logging
from django.http import Http404
from django.core.exceptions import PermissionDenied
from rest_framework import status
from rest_framework.exceptions import (
    APIException,
    AuthenticationFailed,
    NotAuthenticated,
    PermissionDenied as DRFPermissionDenied,
    Throttled,
    ValidationError,
)
from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger("apps.analysis")


class ModelUnavailableError(Exception):
    """Raised when ML models cannot be loaded or are unavailable (НФТ-14)."""

    def __init__(
        self,
        message: str = (
            "Модели машинного обучения временно недоступны. "
            "Анализ сообщений сейчас невозможен."
        ),
    ):
        super().__init__(message)
        self.message = message


STATUS_CODE_MAP = {
    status.HTTP_400_BAD_REQUEST: "VALIDATION_ERROR",
    status.HTTP_401_UNAUTHORIZED: "AUTHENTICATION_ERROR",
    status.HTTP_403_FORBIDDEN: "PERMISSION_DENIED",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMIT_EXCEEDED",
    status.HTTP_503_SERVICE_UNAVAILABLE: "MODEL_UNAVAILABLE",
}


def _extract_error_message(data) -> str:
    """Extract a concise human-readable error message from DRF error payload."""
    if isinstance(data, str):
        return data
    if isinstance(data, list) and data:
        return _extract_error_message(data[0])
    if isinstance(data, dict):
        if "error" in data and isinstance(data["error"], dict) and "message" in data["error"]:
            return str(data["error"]["message"])
        if "detail" in data:
            return _extract_error_message(data["detail"])
        if "non_field_errors" in data:
            return _extract_error_message(data["non_field_errors"])
        for key, value in data.items():
            msg = _extract_error_message(value)
            if msg:
                return msg
    return "Ошибка обработки запроса."


def custom_exception_handler(exc, context):
    """
    Return all API errors in the unified format:
    {"error": {"code": "...", "message": "..."}}
    """
    # Check if it's our ModelUnavailableError or any class named ModelUnavailableError
    if isinstance(exc, ModelUnavailableError) or exc.__class__.__name__ == "ModelUnavailableError":
        msg = getattr(exc, "message", None) or str(exc) or (
            "Модели машинного обучения не загружены. Попробуйте позже."
        )
        logger.error("ModelUnavailableError during API request: %s", msg)
        return Response(
            {
                "error": {
                    "code": "MODEL_UNAVAILABLE",
                    "message": msg,
                }
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    response = exception_handler(exc, context)

    if response is None:
        if isinstance(exc, Http404):
            return Response(
                {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Запрашиваемая запись не найдена.",
                    }
                },
                status=status.HTTP_404_NOT_FOUND,
            )
        if isinstance(exc, PermissionDenied):
            return Response(
                {
                    "error": {
                        "code": "PERMISSION_DENIED",
                        "message": "Доступ запрещён.",
                    }
                },
                status=status.HTTP_403_FORBIDDEN,
            )
        logger.exception("Unhandled API exception: %s", exc.__class__.__name__)
        return Response(
            {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Внутренняя ошибка сервера.",
                }
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Convert SessionAuthentication 403 on unauthenticated requests to 401 when appropriate
    if isinstance(exc, (NotAuthenticated, AuthenticationFailed)):
        response.status_code = status.HTTP_401_UNAUTHORIZED

    status_code = response.status_code
    error_code = STATUS_CODE_MAP.get(status_code, "API_ERROR")

    if isinstance(exc, Throttled):
        error_code = "RATE_LIMIT_EXCEEDED"
        wait = int(exc.wait) if exc.wait else 60
        message = f"Превышен лимит запросов (не более 30 проверок в минуту). Повторите через {wait} с."
    elif isinstance(exc, (NotAuthenticated, AuthenticationFailed)):
        error_code = "AUTHENTICATION_ERROR"
        message = _extract_error_message(response.data)
        if message in ("Учетные данные не были предоставлены.", "Authentication credentials were not provided."):
            message = "Требуется авторизация для выполнения запроса."
    elif isinstance(exc, DRFPermissionDenied):
        error_code = "PERMISSION_DENIED"
        message = _extract_error_message(response.data)
    elif status_code == status.HTTP_404_NOT_FOUND:
        error_code = "NOT_FOUND"
        message = "Запрашиваемая запись не найдена."
    elif isinstance(exc, ValidationError):
        error_code = "VALIDATION_ERROR"
        message = _extract_error_message(response.data)
    else:
        message = _extract_error_message(response.data)

    # If the view already returned {"error": {"code": ..., "message": ...}}, preserve it
    if (
        isinstance(response.data, dict)
        and "error" in response.data
        and isinstance(response.data["error"], dict)
        and "code" in response.data["error"]
        and "message" in response.data["error"]
    ):
        return response

    error_payload = {
        "error": {
            "code": error_code,
            "message": message,
        }
    }
    if isinstance(exc, ValidationError) and isinstance(response.data, dict):
        error_payload["error"]["fields"] = response.data

    response.data = error_payload
    return response
