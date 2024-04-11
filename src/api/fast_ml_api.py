import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, List

import hydra
import uvicorn
from conf.cached import Cache
from conf.prometheus_metrics import PrometheusMetrics
from fastapi import Body, FastAPI, Header, HTTPException, Path, Request, status
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               generate_latest)
from starlette import status
from starlette.responses import Response
from utility.constants import JsonRequestKeys, SignatureColumn
from utility.rest_util import (convert_request_body, convert_request_signature,
                               mailtext2json, convert_request_email_metadata)
from utility.schemas import SignatureBlockResponse, SignatureResponse
from utility.signature_segmentation_util import (get_named_entities,
                                                 get_signature, populate_signature_response)

req_metrics = PrometheusMetrics()
app = FastAPI()
# make this only for dev environment but for production list the allowed origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


@app.get('/')
async def default():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}


@app.get('/liveness')
async def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {'status': 'live'}


@app.get("/readiness")
async def readiness():
    """
    Returns a 200 if the server is ready for prediction.
    """

    # Currently simply take the first tenant.
    # to decrease chances of loading a not needed model.
    return {'status': 'ready'}


@app.get('/metrics')
async def metrics():
    # data, response_headers = get_metrics()
    # registry = REGISTRY
    data = generate_latest(req_metrics.registry)
    response_headers = {
        'Content-type': CONTENT_TYPE_LATEST,
        'Content-Length': str(len(data))
    }
    return Response(data, status_code=status.HTTP_200_OK, headers=response_headers)


async def get_data_from_file(file_path: str) -> Generator:
    with open(file=file_path, mode="rb") as file_like:
        yield file_like.read()


@app.post("/signature/{language}", response_model=SignatureBlockResponse)
async def signature(
        language: str = Path(title="Language"),
        body: str = Body(..., media_type='text/plain')
):
    try:
        with req_metrics.counter_exception.count_exceptions():
            language = language_map.get(language, language)
            json_data = json.loads(mailtext2json(body))
            text = json_data[JsonRequestKeys.data.value][SignatureColumn.body.value]
            return get_signature(text, language)
    except Exception as e:
        return_message = f"Signature extraction failed with error: {e}"
        log_message = f"{return_message} for input: {body} and language: {language}"
        log.error(log_message, exc_info=True)
        raise HTTPException(
            status_code=404, detail=return_message)


@app.post("/signature/ner/{language}")
async def signature_ner(
        language: str = Path(title="Language"),
        raw: bool = True,
        body: str = Body(..., media_type='text/plain')
):
    try:
        with req_metrics.counter_exception.count_exceptions():
            language = language_map.get(language, language)
            json_data = json.loads(mailtext2json(body))
            text = json_data[JsonRequestKeys.data.value][SignatureColumn.body.value]
            response = get_named_entities(text, language, None, not raw)
            return response
    except Exception as e:
        return_message = f"Signature NER failed with error: {e}"
        log_message = f"{return_message} for input: {body} and language: {language}"
        log.error(log_message, exc_info=True)
        raise HTTPException(
            status_code=404, detail=return_message)


@app.post("/v1.0/parse/body/{language}", response_model=List[SignatureResponse])
async def parse_body(
        language: str = Path(title="Language"),
        body: dict = Body(..., media_type='application/json')
):
    log.info(f"Language: {language}")
    try:
        with req_metrics.counter_exception.count_exceptions():
            language = language_map.get(language, language)
            converted_json = convert_request_body(body)
            json_request_metadata = convert_request_email_metadata(body)
            text = converted_json["data"]["ndarray"][0]
            _, signature_text, _, _, _ = get_signature(text, language, False)
            if signature_text and len(signature_text) > 0:
                json_data = json.loads(mailtext2json(signature_text))
                signature_text = json_data[JsonRequestKeys.data.value][SignatureColumn.body.value]
                response = get_named_entities(
                    signature_text, language, json_request_metadata, True)
            else:
                response = populate_signature_response(
                    entities={}, signature_block='', json_request_metadata=json_request_metadata)
            return response
    except Exception as e:
        return_message = f"Signature extraction /v1.0/parse/body/{language} failed with error: {e}"
        log_message = f"{return_message} for input: {body} and language: {language}"
        log.error(log_message, exc_info=True)
        raise HTTPException(
            status_code=404, detail=return_message)


@app.post("/v1.0/parse/signature/{language}", response_model=SignatureResponse)
async def parse_signature(
        language: str = Path(title="Language"),
        body: dict = Body(..., media_type='application/json')
):
    log.info(f"Language: {language}")
    try:
        with req_metrics.counter_exception.count_exceptions():
            language = language_map.get(language, language)
            json_data = json.loads(mailtext2json(
                convert_request_signature(body)))
            signature_text = json_data[JsonRequestKeys.data.value][SignatureColumn.body.value]
            json_request_metadata = convert_request_email_metadata(body)

            if signature_text and len(signature_text) > 0:
                response = get_named_entities(
                    signature_text, language, json_request_metadata, True)
            else:
                response = populate_signature_response(
                    entities={}, signature_block='', json_request_metadata=json_request_metadata)
            return response[0]
    except Exception as e:
        return_message = f"Signature extraction /v1.0/parse/signature/{language} failed with error: {e}"
        log_message = f"{return_message} for input: {signature_text} and language: {language}"
        log.error(log_message, exc_info=True)
        raise HTTPException(
            status_code=404, detail=return_message)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if 'startup_pipelines' in cfg and os.getenv("bypass_startup_load", "false") == "false":
        for pipeline in cfg.startup_pipelines:
            log.info(f"Starting pipeline: {pipeline}")
            Cache.pipelines[pipeline]
    if os.environ.get('PORT') is not None:
        cfg.server.port = os.environ["PORT"]
    log.info(
        f"********************* Ready to Serve on port {cfg.server.port} ********************")
    uvicorn.run(app, **cfg.server)
    registry = CollectorRegistry(True)
    request_metrics = Counter("http_exception",
                              "Number of exceptions events")
    registry.register(request_metrics)


if __name__ == "__main__":
    main()