import json
from PIL import Image
from http import HTTPStatus
from .reader_model import BrailleRecognizer
from rest_framework.decorators import api_view
from django.http import HttpResponse, HttpRequest


@api_view(["GET"])
def say_hello(request: HttpRequest):
    return HttpResponse("Hello World")


@api_view(["POST"])
def transcribe_braille(request: HttpRequest):
    if request.method == "POST":
        try:
            image = request.FILES["image"]
            image = Image.open(image)
            br = BrailleRecognizer()
            res: dict = br.recognizer.recognize(
                image,
                find_orientation=True,
                align_results=True,
            )
            return HttpResponse(res, status=HTTPStatus.OK)
        except Exception as e:
            return HttpResponse(e, status=HTTPStatus.BAD_REQUEST)
    else:
        return HttpResponse("Error", status=HTTPStatus.BAD_REQUEST)


@api_view(["POST"])
def test(request: HttpRequest):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            image = body["image"]
            return HttpResponse(image, status=HTTPStatus.OK)
        except Exception as e:
            return HttpResponse(e, status=HTTPStatus.BAD_REQUEST)
    else:
        return HttpResponse("Error", status=HTTPStatus.BAD_REQUEST)
