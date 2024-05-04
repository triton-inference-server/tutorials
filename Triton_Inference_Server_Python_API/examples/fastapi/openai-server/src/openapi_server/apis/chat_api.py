# coding: utf-8

import importlib
import pkgutil
from typing import Dict, List  # noqa: F401

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    Path,
    Query,
    Response,
    Security,
    status,
)

import openapi_server.impl
from openapi_server.apis.chat_api_base import BaseChatApi
from openapi_server.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from openapi_server.models.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from openapi_server.models.extra_models import TokenModel  # noqa: F401
from openapi_server.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = openapi_server.impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/chat/completions",
    responses={
        200: {"model": CreateChatCompletionResponse, "description": "OK"},
    },
    tags=["Chat"],
    summary="Creates a model response for the given chat conversation.",
    response_model_by_alias=True,
)
async def create_chat_completion(
    create_chat_completion_request: CreateChatCompletionRequest = Body(
        None, description=""
    ),
    token_ApiKeyAuth: TokenModel = Security(get_token_ApiKeyAuth),
) -> CreateChatCompletionResponse:
    ...
