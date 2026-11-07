from django.urls import path
from . import views

urlpatterns = [
    path("summary/", views.StatsSummaryAPIView.as_view(), name="api_stats_summary"),
]
