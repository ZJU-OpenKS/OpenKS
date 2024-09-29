from fastapi import FastAPI
from starlette_prometheus import PrometheusMiddleware, metrics

from .routers.ner import router as ner_router

app = FastAPI()

# prometheus
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# routers
app.include_router(ner_router, prefix="/nlp/ner", tags=["ner"])
