# coding: utf-8

import copy
import importlib
import pkgutil
import time
import uuid
from typing import Dict, List  # noqa: F401

import openapi_server.impl
import tritonserver
from openapi_server.apis.completions_api_base import BaseCompletionsApi

triton_server = tritonserver.Server(
    model_repository="/workspace/llm-models", log_verbose=6, strict_model_config=False
).start()


from fastapi import (  # noqa: F401; Security,
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    Path,
    Query,
    Response,
    status,
)
from openapi_server.models.create_completion_request import CreateCompletionRequest
from openapi_server.models.create_completion_response import CreateCompletionResponse
from openapi_server.models.create_completion_response_choices_inner import (
    CreateCompletionResponseChoicesInner,
)
from openapi_server.models.create_completion_response_choices_inner_logprobs import (
    CreateCompletionResponseChoicesInnerLogprobs,
)
from openapi_server.models.extra_models import TokenModel  # noqa: F401

# from openapi_server.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = openapi_server.impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/completions",
    responses={
        200: {"model": CreateCompletionResponse, "description": "OK"},
    },
    tags=["Completions"],
    summary="Creates a completion for the provided prompt and parameters.",
    response_model_by_alias=True,
)
async def create_completion(
    create_completion_request: CreateCompletionRequest = Body(None, description=""),
) -> CreateCompletionResponse:
    exclude_input_in_output = True

    if create_completion_request.echo:
        exclude_input_in_output = False

    model = triton_server.model("llama-3-8b-instruct")
    parameters = copy.deepcopy(create_completion_request.dict())
    del parameters["prompt"]
    del parameters["stream"]
    del parameters["echo"]
    del parameters["model"]

    response = list(
        model.infer(
            inputs={
                "text_input": [create_completion_request.prompt],
                "stream": [False],
                "exclude_input_in_output": [exclude_input_in_output],
            },
            parameters=parameters,
        )
    )[0]

    choice = CreateCompletionResponseChoicesInner(
        finish_reason="stop",
        index=0,
        text=response.outputs["text_output"].to_string_array()[0],
        logprobs=CreateCompletionResponseChoicesInnerLogprobs(),
    )

    return CreateCompletionResponse(
        id=f"cmpl-{uuid.uuid1()}",
        created=int(time.time()),
        model=model.name,
        choices=[choice],
        system_fingerprint=None,
        object="text_completion",
    )
