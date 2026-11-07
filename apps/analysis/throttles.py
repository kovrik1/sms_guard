"""
Rate limiting (throttling) for SMS and URL check endpoints (НФТ-6: max 30 checks/min per user).
"""
from rest_framework.throttling import UserRateThrottle


class CheckUserRateThrottle(UserRateThrottle):
    """Enforce at most 30 check requests per minute per authenticated user."""

    scope = "check_rate"
    rate = "30/min"
