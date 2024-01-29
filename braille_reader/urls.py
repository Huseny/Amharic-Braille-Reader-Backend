from django.urls import path
from . import views

urlpatterns = [
    path("", views.say_hello),
    path("upload_image/", views.transcribe_braille),
    path("test/", views.test),
]