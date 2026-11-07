from django.urls import path
from . import views

app_name = "analysis"

urlpatterns = [
    path("check/sms/", views.check_sms_view, name="check_sms"),
    path("check/url/", views.check_url_view, name="check_url"),
    path("checks/<int:pk>/", views.check_detail_view, name="check_detail"),
    path("checks/<int:pk>/delete/", views.check_delete_view, name="check_delete"),
    path("history/", views.history_view, name="history"),
    path("history/export/", views.history_export_csv_view, name="history_export"),
]
