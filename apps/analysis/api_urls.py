from django.urls import path
from . import views

urlpatterns = [
    path("check/sms/", views.CheckSmsAPIView.as_view(), name="api_check_sms"),
    path("check/url/", views.CheckUrlAPIView.as_view(), name="api_check_url"),
    path("checks/", views.CheckListAPIView.as_view(), name="api_check_list"),
    path(
        "checks/<int:pk>/",
        views.CheckDetailDeleteAPIView.as_view(),
        name="api_check_detail",
    ),
    path("health/", views.HealthAPIView.as_view(), name="api_health"),
]
