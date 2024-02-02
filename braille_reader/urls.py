from django.urls import path
from . import views

urlpatterns = [
    path("", views.say_hello),
    path("upload_image/", views.transcribe_braille),
    path("get_audio/<str:id>", views.get_audio),
    path("test/", views.test),
]