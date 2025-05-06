from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('api/', include('predictor_new.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)