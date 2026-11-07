"""
Root URL configuration for SMS Guard project.
"""
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views.static import serve
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path("admin/", admin.site.urls),
    # OpenAPI / Swagger documentation (Задание 14 п. 8)
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    # REST API v1 (Раздел 4.4)
    path("api/v1/auth/", include("apps.accounts.api_urls")),
    path("api/v1/", include("apps.analysis.api_urls")),
    path("api/v1/stats/", include("apps.dashboard.api_urls")),
    # Web interface (Раздел 8.1)
    path("", include("apps.accounts.urls")),
    path("", include("apps.dashboard.urls")),
    path("", include("apps.analysis.urls")),
]

# Serve static files even when DEBUG=False (for demonstration mode НФТ-7)
if settings.STATICFILES_DIRS:
    urlpatterns += [
        re_path(
            r"^static/(?P<path>.*)$",
            serve,
            {"document_root": settings.STATICFILES_DIRS[0]},
        ),
    ]
