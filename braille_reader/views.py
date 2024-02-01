import json
import uuid
from PIL import Image
from http import HTTPStatus
from rest_framework.decorators import api_view
from django.http import HttpResponse, HttpRequest
from braille_reader.reader_model.main import BrailleRecognizer


@api_view(["GET"])
def say_hello(request: HttpRequest):
    return HttpResponse("Hello World")


@api_view(["POST"])
def transcribe_braille(request: HttpRequest):
    if request.method == "POST":
        try:
            image = request.FILES["image"]
            image = Image.open(image)
            id = str(uuid.uuid4())
            br = BrailleRecognizer()
            res: dict = br.recognize(
                image,
                find_orientation=True,
                align_results=True,
            )
            result = {
                "id": str(id),
                "braille": res["braille"],
                "translation": res["text"],
            }
            return HttpResponse(
                json.dumps(result),
                content_type="application/json",
                status=HTTPStatus.OK,
            )
        except Exception as e:
            print("Error: ", e)
            return HttpResponse(e, status=HTTPStatus.BAD_REQUEST)
    else:
        print("Error: ", e)
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
