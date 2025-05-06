from django.urls import path
from . import views

urlpatterns = [
    path('options/', views.get_options, name='get_options'),
    path('predict/', views.predict_match, name='predict_match'),
    path('chat/', views.chat_with_model, name='chat_with_model'),  # Added new endpoint
]