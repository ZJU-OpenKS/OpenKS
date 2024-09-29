import ujson
from fastapi import APIRouter, Response
from pydantic import BaseModel

from backend.NER.predict import predict

router = APIRouter()

predictor = predict()


class NERRequest(BaseModel):
    text: str


@router.post("", tags=["ner"])
@router.post("/", tags=["ner"])
async def ner(request: NERRequest):
    ners = predictor.predict(request.text)
    r = {"text": request.text, "nes": ners}
    content = ujson.dumps(r, ensure_ascii=False).encode("utf-8")
    return Response(content=content, media_type="application/json")
