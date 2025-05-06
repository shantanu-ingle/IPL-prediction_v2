from django.urls import path
from . import views

urlpatterns = [
    path('options/', views.get_options, name='get_options'),
    path('predict/', views.predict_match, name='predict_match'),
]