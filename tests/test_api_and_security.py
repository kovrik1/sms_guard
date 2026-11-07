"""
Комплексный набор интеграционных, функциональных автотестов и тестов безопасности
для Заданий 12, 14, 15, 16, 17, 18 (ФТ-1.1..ФТ-3.11, НФТ-1..НФТ-14).
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client
from django.utils import timezone
from rest_framework.test import APIClient

from apps.accounts.models import UserProfile, get_or_create_profile
from apps.analysis import services as analysis_services
from apps.analysis.models import Check, DetectedFeature, UrlResult
from config.exceptions import ModelUnavailableError

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def clear_django_cache():
    """Очистка кэша перед каждым тестом (сброс счётчиков throttling и блокировок входа)."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def api_client() -> APIClient:
    return APIClient()


@pytest.fixture
def user_a() -> User:
    user = User.objects.create_user(
        username="user_alpha",
        email="alpha@example.com",
        password="SafePassword123",
    )
    get_or_create_profile(user)
    return user


@pytest.fixture
def user_b() -> User:
    user = User.objects.create_user(
        username="user_bravo",
        email="bravo@example.com",
        password="SafePassword123",
    )
    get_or_create_profile(user)
    return user


@pytest.fixture
def admin_user() -> User:
    user = User.objects.create_user(
        username="admin_staff",
        email="admin@example.com",
        password="AdminPassword123",
        is_staff=True,
    )
    get_or_create_profile(user)
    return user


@pytest.fixture
def auth_client_a(user_a: User) -> APIClient:
    client = APIClient()
    client.force_authenticate(user=user_a)
    return client


@pytest.fixture
def auth_client_b(user_b: User) -> APIClient:
    client = APIClient()
    client.force_authenticate(user=user_b)
    return client


# ==============================================================================
# 1. Аутентификация и защита сессий (ФТ-1.1 – ФТ-1.9)
# ==============================================================================


def test_01_register_valid_user_creates_profile(api_client: APIClient):
    """ФТ-1.1, ФТ-1.4: Регистрация валидного пользователя создаёт UserProfile(store_sms_text=False) и хеширует пароль."""
    resp = api_client.post(
        "/api/v1/auth/register/",
        {
            "username": "new_student_1",
            "email": "student1@example.com",
            "password": "StrongPass123",
        },
        format="json",
    )
    assert resp.status_code == 201
    assert resp.data["username"] == "new_student_1"

    user = User.objects.get(username="new_student_1")
    assert user.password.startswith("pbkdf2_sha256$")
    assert "StrongPass123" not in user.password

    profile = UserProfile.objects.get(user=user)
    assert profile.store_sms_text is False


def test_02_register_weak_password_and_duplicate_username_rejected(
    api_client: APIClient, user_a: User
):
    """ФТ-1.1, ФТ-1.2, НФТ-11: Слабый пароль и повтор логина возвращают 400 в едином формате ошибок."""
    # Слабый пароль (только цифры или < 8 символов)
    resp_weak = api_client.post(
        "/api/v1/auth/register/",
        {
            "username": "valid_user_2",
            "email": "valid2@example.com",
            "password": "short",
        },
        format="json",
    )
    assert resp_weak.status_code == 400
    assert "error" in resp_weak.data
    assert resp_weak.data["error"]["code"] == "VALIDATION_ERROR"
    assert resp_weak.data["error"]["message"]

    # Дубликат логина
    resp_dup = api_client.post(
        "/api/v1/auth/register/",
        {
            "username": user_a.username,
            "email": "another@example.com",
            "password": "SafePassword123",
        },
        format="json",
    )
    assert resp_dup.status_code == 400
    assert "error" in resp_dup.data
    assert resp_dup.data["error"]["code"] == "VALIDATION_ERROR"


def test_03_login_and_logout_lifecycle(api_client: APIClient, user_a: User):
    """ФТ-1.3, ФТ-1.5: Успешный вход устанавливает сессию (200), выход уничтожает её (204)."""
    login_resp = api_client.post(
        "/api/v1/auth/login/",
        {"username": user_a.username, "password": "SafePassword123"},
        format="json",
    )
    assert login_resp.status_code == 200
    assert login_resp.data["username"] == user_a.username

    logout_resp = api_client.post("/api/v1/auth/logout/")
    assert logout_resp.status_code in (200, 204)

    # После выхода защищённый эндпоинт недоступен
    checks_resp = api_client.get("/api/v1/checks/")
    assert checks_resp.status_code in (401, 403)


def test_04_login_brute_force_lockout_after_5_attempts(
    api_client: APIClient, user_a: User
):
    """ФТ-1.3: После 5 неудачных попыток входа подряд 6-я попытка блокируется на 60 с (429)."""
    for _ in range(5):
        r = api_client.post(
            "/api/v1/auth/login/",
            {"username": user_a.username, "password": "WrongPassword999"},
            format="json",
        )
        assert r.status_code == 401

    # 6-я попытка (даже с верным паролем) во время блокировки возвращает 429
    r_locked = api_client.post(
        "/api/v1/auth/login/",
        {"username": user_a.username, "password": "SafePassword123"},
        format="json",
    )
    assert r_locked.status_code == 429
    assert r_locked.data["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_05_unauthenticated_access_redirects_html_and_blocks_api(
    api_client: APIClient, client: Client
):
    """ФТ-1.6: Неавторизованный доступ к HTML-страницам редиректит на /login/ (302), к API — 401/403."""
    for page_url in ("/check/sms/", "/check/url/", "/history/", "/stats/", "/profile/"):
        html_resp = client.get(page_url)
        assert html_resp.status_code == 302
        assert "/login/" in html_resp["Location"]

    api_resp = api_client.get("/api/v1/checks/")
    assert api_resp.status_code in (401, 403)


def test_06_cookie_flags_and_csrf_protection(user_a: User):
    """ФТ-1.8: Проверка настроек флагов cookie (HttpOnly, SameSite=Lax) и CSRF-защиты POST-форм."""
    assert settings.SESSION_COOKIE_HTTPONLY is True
    assert settings.SESSION_COOKIE_SAMESITE == "Lax"
    assert settings.CSRF_COOKIE_SAMESITE == "Lax"

    csrf_client = Client(enforce_csrf_checks=True)
    # POST без CSRF-токена отклоняется с кодом 403
    resp = csrf_client.post(
        "/login/",
        {"username": user_a.username, "password": "SafePassword123"},
    )
    assert resp.status_code == 403


# ==============================================================================
# 2. Проверка SMS и URL, дедупликация и хранение истории (ФТ-2, ФТ-3, Задания 12, 14, 16)
# ==============================================================================


def test_07_health_endpoint_returns_models_status(api_client: APIClient):
    """Задание 2 п. 8, Раздел 4.4: GET /api/v1/health/ возвращает статус приложения и версий моделей."""
    resp = api_client.get("/api/v1/health/")
    assert resp.status_code == 200
    assert resp.data["status"] == "ok"
    assert "models_loaded" in resp.data
    assert "model_versions" in resp.data
    assert isinstance(resp.data["model_versions"], dict)


def test_08_check_safe_sms_and_fraudulent_sms_with_url(auth_client_a: APIClient, user_a: User):
    """ФТ-2.2..ФТ-2.7, Задание 12: Анализ безопасного и мошеннического SMS с извлечением ссылки."""
    # 1. Безопасное сообщение
    safe_resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": "Привет, встречаемся завтра у входа в библиотеку."},
        format="json",
    )
    assert safe_resp.status_code == 200
    assert safe_resp.data["type"] == "SMS"
    assert safe_resp.data["language"] == "ru"
    assert safe_resp.data["category"] == "SAFE"
    assert 0.0 <= safe_resp.data["p_sms"] <= 1.0
    assert safe_resp.data["urls"] == []
    assert safe_resp.data["disclaimer"]

    # 2. Мошенническое сообщение со ссылкой и социальной инженерией
    fraud_resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {
            "text": (
                "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: "
                "http://secure-bank-verify.example/login"
            )
        },
        format="json",
    )
    assert fraud_resp.status_code == 200
    assert fraud_resp.data["category"] in ("SUSPICIOUS", "FRAUDULENT")
    assert fraud_resp.data["risk_score"] >= 50
    assert len(fraud_resp.data["urls"]) == 1
    assert fraud_resp.data["urls"][0]["domain"] == "secure-bank-verify.example"
    assert len(fraud_resp.data["features"]) >= 1
    assert len(fraud_resp.data["explanation"]) >= 1
    assert fraud_resp.data["recommendation"]

    check_obj = Check.objects.get(pk=fraud_resp.data["id"])
    assert UrlResult.objects.filter(check=check_obj).count() == 1
    assert DetectedFeature.objects.filter(check=check_obj).count() >= 1


def test_09_privacy_store_sms_text_consent_and_masking(
    auth_client_a: APIClient, user_a: User
):
    """Раздел 2.7, Задание 16 п. 4: Без согласия input_text=None и данные маскируются '***'; с согласием — сохраняются."""
    profile = get_or_create_profile(user_a)
    assert profile.store_sms_text is False

    sms_with_secrets = "Переведите 25000 ₸ на карту 4400430011112222 или +77012345678"
    resp1 = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": sms_with_secrets},
        format="json",
    )
    assert resp1.status_code == 200
    check1 = Check.objects.get(pk=resp1.data["id"])
    assert check1.input_text is None
    assert "***" in check1.input_masked
    assert "4400430011112222" not in check1.input_masked

    # Включаем согласие на хранение текста в профиле
    profile.store_sms_text = True
    profile.save(update_fields=["store_sms_text"])

    sms_with_consent = "Другой текст проверки с суммой 15000 ₸ для сохранения"
    resp2 = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": sms_with_consent},
        format="json",
    )
    assert resp2.status_code == 200
    check2 = Check.objects.get(pk=resp2.data["id"])
    assert check2.input_text == sms_with_consent


def test_10_duplicate_submission_prevention_on_double_click(
    auth_client_a: APIClient, user_a: User
):
    """НФТ-3: Повторная отправка того же SMS в течение окна дедупликации не создаёт вторую запись в БД."""
    payload = {"text": "Уникальное сообщение для проверки защиты от двойного клика"}
    r1 = auth_client_a.post("/api/v1/check/sms/", payload, format="json")
    r2 = auth_client_a.post("/api/v1/check/sms/", payload, format="json")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.data["id"] == r2.data["id"]
    assert Check.objects.filter(user=user_a).count() == 1


def test_11_check_url_safe_and_dangerous_and_invalid_scheme(auth_client_a: APIClient):
    """ФТ-3.1..ФТ-3.9: Проверка безопасного и опасного URL, отклонение ftp:// и URL без домена (400)."""
    safe_url_resp = auth_client_a.post(
        "/api/v1/check/url/",
        {"url": "https://www.example.com/about"},
        format="json",
    )
    assert safe_url_resp.status_code == 200
    assert safe_url_resp.data["type"] == "URL"
    assert safe_url_resp.data["category"] == "SAFE"

    evil_url_resp = auth_client_a.post(
        "/api/v1/check/url/",
        {"url": "http://192.0.2.15/secure/login?verify=1"},
        format="json",
    )
    assert evil_url_resp.status_code == 200
    assert evil_url_resp.data["category"] in ("SUSPICIOUS", "DANGEROUS")
    assert evil_url_resp.data["risk_score"] >= 60

    # Недопустимая схема ftp:// и URL без домена
    for bad_url in ("ftp://files.example.com/archive.zip", "http://", ":///no-domain"):
        bad_resp = auth_client_a.post(
            "/api/v1/check/url/",
            {"url": bad_url},
            format="json",
        )
        assert bad_resp.status_code == 400
        assert bad_resp.data["error"]["code"] == "VALIDATION_ERROR"


def test_12_checks_history_pagination_and_filters(
    auth_client_a: APIClient, user_a: User
):
    """Раздел 2.7, Задание 16 п. 5: Пагинация по 20 записей и фильтрация по type, category, risk_level, date_from/to."""
    now = timezone.now()
    for i in range(23):
        is_fraud = i % 2 == 0
        Check.objects.create(
            user=user_a,
            check_type=Check.TYPE_SMS if i < 20 else Check.TYPE_URL,
            input_masked=f"Тестовая запись #{i}",
            language="ru",
            category=Check.CATEGORY_FRAUDULENT if is_fraud else Check.CATEGORY_SAFE,
            risk_level=Check.RISK_HIGH if is_fraud else Check.RISK_LOW,
            risk_score=85 if is_fraud else 10,
            p_sms=0.9 if is_fraud else 0.05,
            submission_hash=f"hash_{i}",
        )

    # Страница 1 содержит 20 записей, страница 2 — 3 записи
    p1 = auth_client_a.get("/api/v1/checks/?page=1")
    assert p1.status_code == 200
    assert p1.data["count"] == 23
    assert len(p1.data["results"]) == 20

    p2 = auth_client_a.get("/api/v1/checks/?page=2")
    assert p2.status_code == 200
    assert len(p2.data["results"]) == 3

    # Фильтр по типу URL и категории
    filtered = auth_client_a.get(
        f"/api/v1/checks/?type=URL&category=FRAUDULENT&risk_level=HIGH&date_from={(now - timedelta(days=1)).strftime('%Y-%m-%d')}"
    )
    assert filtered.status_code == 200
    for item in filtered.data["results"]:
        assert item["type"] == "URL"
        assert item["category"] == "FRAUDULENT"
        assert item["risk_level"] == "HIGH"


def test_13_history_csv_export_contains_only_own_records(
    client: Client, user_a: User, user_b: User
):
    """Раздел 2.7: GET /history/export/ выгружает CSV только с записями текущего пользователя."""
    Check.objects.create(
        user=user_a,
        check_type=Check.TYPE_SMS,
        input_masked="Запись пользователя А",
        language="ru",
        category=Check.CATEGORY_SAFE,
        risk_level=Check.RISK_LOW,
        risk_score=5,
    )
    Check.objects.create(
        user=user_b,
        check_type=Check.TYPE_SMS,
        input_masked="Чужая секретная запись Б",
        language="ru",
        category=Check.CATEGORY_FRAUDULENT,
        risk_level=Check.RISK_HIGH,
        risk_score=90,
    )

    client.force_login(user_a)
    resp = client.get("/history/export/")
    assert resp.status_code == 200
    assert "text/csv" in resp["Content-Type"]
    body = resp.content.decode("utf-8-sig")
    assert "Запись пользователя А" in body
    assert "Чужая секретная запись Б" not in body


def test_14_delete_own_check_record(auth_client_a: APIClient, user_a: User):
    """Раздел 4.4, Задание 16 п. 6: DELETE /api/v1/checks/{id}/ удаляет запись владельца (204)."""
    check = Check.objects.create(
        user=user_a,
        check_type=Check.TYPE_SMS,
        input_masked="Удаляемая запись",
        language="ru",
        category=Check.CATEGORY_SAFE,
        risk_level=Check.RISK_LOW,
        risk_score=10,
    )
    resp = auth_client_a.delete(f"/api/v1/checks/{check.id}/")
    assert resp.status_code == 204
    assert not Check.objects.filter(pk=check.id).exists()


# ==============================================================================
# 3. Безопасность и изоляция данных (Задание 18, НФТ-4, НФТ-5, НФТ-6, НФТ-9, ФТ-1.7, ФТ-1.9)
# ==============================================================================


def test_15_idor_protection_foreign_check_returns_404(
    auth_client_b: APIClient, client: Client, user_a: User, user_b: User
):
    """ФТ-1.7: Попытка пользователя B просмотреть или удалить проверку пользователя A возвращает 404."""
    check_a = Check.objects.create(
        user=user_a,
        check_type=Check.TYPE_SMS,
        input_masked="Приватная проверка пользователя А",
        language="ru",
        category=Check.CATEGORY_FRAUDULENT,
        risk_level=Check.RISK_HIGH,
        risk_score=88,
    )

    # API GET и DELETE от имени user_b
    get_resp = auth_client_b.get(f"/api/v1/checks/{check_a.id}/")
    assert get_resp.status_code == 404

    del_resp = auth_client_b.delete(f"/api/v1/checks/{check_a.id}/")
    assert del_resp.status_code == 404

    # HTML-страница деталей от имени user_b
    client.force_login(user_b)
    html_resp = client.get(f"/checks/{check_a.id}/")
    assert html_resp.status_code == 404

    # Запись пользователя A осталась в БД
    assert Check.objects.filter(pk=check_a.id).exists()


def test_16_sql_injection_payloads_handled_safely(
    auth_client_a: APIClient, user_b: User
):
    """НФТ-4: SQL-инъекции в тексте SMS и фильтрах истории не вызывают 500 и не раскрывают чужие записи."""
    Check.objects.create(
        user=user_b,
        check_type=Check.TYPE_SMS,
        input_masked="Чужая запись",
        language="ru",
        category=Check.CATEGORY_SAFE,
        risk_level=Check.RISK_LOW,
        risk_score=10,
    )

    sqli_text = "' OR 1=1 --; DROP TABLE analysis_check; --"
    sms_resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": sqli_text},
        format="json",
    )
    assert sms_resp.status_code == 200

    filter_resp = auth_client_a.get(
        "/api/v1/checks/",
        {"category": "' OR 1=1 --", "type": "1 UNION SELECT password FROM auth_user"},
    )
    assert filter_resp.status_code == 200
    # Чужая запись user_b не попала в выборку user_a
    returned_ids = {item["id"] for item in filter_resp.data["results"]}
    assert not Check.objects.filter(user=user_b, id__in=returned_ids).exists()


def test_17_xss_payload_escaped_in_html_and_non_clickable_urls(
    auth_client_a: APIClient, client: Client, user_a: User
):
    """НФТ-5, ФТ-3.10: Тег <script> экранируется на странице результата, а опасная ссылка не является гиперссылкой."""
    xss_sms = "<script>alert(1)</script> Перейдите на http://secure-bank-verify.example/login"
    api_resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": xss_sms},
        format="json",
    )
    assert api_resp.status_code == 200
    check_id = api_resp.data["id"]

    client.force_login(user_a)
    detail_resp = client.get(f"/checks/{check_id}/")
    assert detail_resp.status_code == 200
    html = detail_resp.content.decode("utf-8")

    # Сырой <script>alert(1)</script> не должен присутствовать в HTML, только экранированный вариант
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # Ссылка не должна рендериться как кликабельный <a href="http://secure-bank-verify.example/login">
    assert 'href="http://secure-bank-verify.example/login"' not in html


def test_18_rate_limiting_30_requests_per_minute(
    auth_client_a: APIClient, monkeypatch
):
    """НФТ-6: Ограничение частоты API — не более 30 проверок в минуту на пользователя (31-й запрос -> 429)."""
    fake_result = analysis_services.AnalysisResult(
        type=Check.TYPE_SMS,
        language="ru",
        category="SAFE",
        risk_level="LOW",
        risk_score=5,
        p_sms=0.05,
        p_url_max=None,
    )
    monkeypatch.setattr("apps.analysis.views.analyze_sms", lambda text: fake_result)

    for i in range(30):
        r = auth_client_a.post(
            "/api/v1/check/sms/",
            {"text": f"Тестовое сообщение для лимита #{i}"},
            format="json",
        )
        assert r.status_code == 200, f"Request #{i + 1} failed with {r.status_code}"

    # 31-й запрос в пределах минуты должен вернуть 429 Too Many Requests
    r_31 = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": "Тридцать первый запрос сверх лимита"},
        format="json",
    )
    assert r_31.status_code == 429
    assert r_31.data["error"]["code"] == "RATE_LIMIT_EXCEEDED"


def test_19_logs_do_not_contain_raw_sms_text(auth_client_a: APIClient):
    """НФТ-9: Тексты SMS никогда не записываются в лог-файл logs/app.log."""
    unique_secret_marker = "TOP_SECRET_SMS_PAYLOAD_994812_DO_NOT_LOG"
    resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": f"Срочно переведите деньги {unique_secret_marker}"},
        format="json",
    )
    assert resp.status_code == 200

    log_file = Path(settings.BASE_DIR) / "logs" / "app.log"
    if log_file.exists():
        log_content = log_file.read_text(encoding="utf-8", errors="ignore")
        assert unique_secret_marker not in log_content


def test_20_admin_stats_summary_aggregates_all_users_without_foreign_sms_text(
    api_client: APIClient, user_a: User, user_b: User, admin_user: User
):
    """ФТ-1.9, Задание 17 п. 8: Администратор видит агрегированную статистику по всем пользователям без чужих текстов SMS."""
    secret_a = "СЕКРЕТНЫЙ_ТЕКСТ_ПОЛЬЗОВАТЕЛЯ_А_7711"
    secret_b = "СЕКРЕТНЫЙ_ТЕКСТ_ПОЛЬЗОВАТЕЛЯ_Б_8822"

    Check.objects.create(
        user=user_a,
        check_type=Check.TYPE_SMS,
        input_text=secret_a,
        input_masked=secret_a,
        language="ru",
        category=Check.CATEGORY_SAFE,
        risk_level=Check.RISK_LOW,
        risk_score=10,
    )
    Check.objects.create(
        user=user_b,
        check_type=Check.TYPE_SMS,
        input_text=secret_b,
        input_masked=secret_b,
        language="ru",
        category=Check.CATEGORY_FRAUDULENT,
        risk_level=Check.RISK_HIGH,
        risk_score=90,
    )

    api_client.force_authenticate(user=admin_user)
    resp = api_client.get("/api/v1/stats/summary/?scope=all&days=30")
    assert resp.status_code == 200
    assert resp.data["is_admin_view"] is True
    assert resp.data["total_checks"] >= 2
    assert "active_models" in resp.data

    raw_json = str(resp.data)
    assert secret_a not in raw_json
    assert secret_b not in raw_json


# ==============================================================================
# 4. Надёжность и граничные значения (НФТ-12, НФТ-14, ФТ-2.1)
# ==============================================================================


def test_21_empty_whitespace_and_1001_chars_rejected_with_400(
    auth_client_a: APIClient,
):
    """ФТ-2.1, НФТ-12: Пустой ввод, строка из пробелов и строка из 1001 символа отклоняются с кодом 400 (не 500)."""
    for invalid_text in ("", "     \t\n   ", "А" * 1001):
        resp = auth_client_a.post(
            "/api/v1/check/sms/",
            {"text": invalid_text},
            format="json",
        )
        assert resp.status_code == 400
        assert resp.data["error"]["code"] == "VALIDATION_ERROR"


def test_22_boundary_1000_chars_and_null_bytes_handled_without_500(
    auth_client_a: APIClient,
):
    """ФТ-2.1, НФТ-12: Ровно 1000 символов проходят успешно (200), а нулевой байт \x00 отклоняется с 400 (не 500)."""
    exact_1000 = "Б" * 1000
    resp_ok = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": exact_1000},
        format="json",
    )
    assert resp_ok.status_code == 200

    resp_null = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": "Текст с нулевым байтом \x00 внутри"},
        format="json",
    )
    assert resp_null.status_code == 400
    assert resp_null.data["error"]["code"] == "VALIDATION_ERROR"


def test_23_missing_models_return_503_service_unavailable(
    auth_client_a: APIClient, monkeypatch
):
    """НФТ-14: При отсутствии загруженных моделей эндпоинты проверки возвращают 503 Service Unavailable."""

    def _raise_unavailable():
        raise ModelUnavailableError(
            "Файлы обученных моделей отсутствуют или не загружены. Анализ временно недоступен."
        )

    monkeypatch.setattr(analysis_services, "_get_predictor", _raise_unavailable)

    sms_resp = auth_client_a.post(
        "/api/v1/check/sms/",
        {"text": "Проверка поведения при отсутствии модели"},
        format="json",
    )
    assert sms_resp.status_code == 503
    assert sms_resp.data["error"]["code"] == "MODEL_UNAVAILABLE"

    url_resp = auth_client_a.post(
        "/api/v1/check/url/",
        {"url": "https://www.example.com"},
        format="json",
    )
    assert url_resp.status_code == 503
    assert url_resp.data["error"]["code"] == "MODEL_UNAVAILABLE"

    # При этом /api/v1/health/ продолжает отвечать 200 с models_loaded=False
    health_resp = auth_client_a.get("/api/v1/health/")
    assert health_resp.status_code == 200
    assert health_resp.data["models_loaded"] is False
