"""Регрессионные тесты безопасности Dashboard (НФТ-5): данные для Chart.js не должны разрывать <script>."""
from __future__ import annotations

import pytest
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import Client

from apps.accounts.models import get_or_create_profile
from apps.analysis.models import Check

pytestmark = pytest.mark.django_db

PAYLOAD = "</script><script>alert(1337)</script>"


@pytest.fixture(autouse=True)
def _clear_cache():
    cache.clear()
    yield
    cache.clear()


def _make_user(username: str, is_staff: bool = False) -> User:
    user = User.objects.create_user(username=username, password="SafePassword123", is_staff=is_staff)
    profile = get_or_create_profile(user)
    profile.store_sms_text = True
    profile.save()
    return user


def _make_check(user: User) -> Check:
    return Check.objects.create(
        user=user,
        check_type=Check.TYPE_SMS,
        input_text=PAYLOAD,
        input_masked=PAYLOAD,
        language="ru",
        category="FRAUDULENT",
        risk_level="HIGH",
        risk_score=90,
        p_sms=0.9,
        explanation=[PAYLOAD],
        recommendation=PAYLOAD,
        submission_hash="x" * 8,
    )


@pytest.mark.parametrize("path", ["/", "/stats/"])
def test_dashboard_pages_do_not_break_out_of_script_tag(path):
    user = _make_user("xss_user")
    _make_check(user)
    client = Client()
    client.force_login(user)
    response = client.get(path)
    assert response.status_code == 200
    body = response.content.decode("utf-8")
    assert PAYLOAD not in body
    assert "<script>alert(1337)" not in body


def test_active_models_block_hidden_for_regular_user():
    user = _make_user("plain_user")
    _make_check(user)
    client = Client()
    client.force_login(user)
    body = client.get("/").content.decode("utf-8")
    assert "Активные версии ML-моделей" not in body


def test_active_models_block_visible_for_admin():
    admin = _make_user("admin_user", is_staff=True)
    _make_check(admin)
    client = Client()
    client.force_login(admin)
    body = client.get("/").content.decode("utf-8")
    assert "Активные версии ML-моделей" in body
