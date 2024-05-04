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
from openapi_server.apis.models_api_base import BaseModelsApi
from openapi_server.models.delete_model_response import DeleteModelResponse
from openapi_server.models.extra_models import TokenModel  # noqa: F401
from openapi_server.models.list_models_response import ListModelsResponse
from openapi_server.models.model import Model
from openapi_server.security_api import get_token_ApiKeyAuth

router = APIRouter()

ns_pkg = openapi_server.impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.delete(
    "/models/{model}",
    responses={
        200: {"model": DeleteModelResponse, "description": "OK"},
    },
    tags=["Models"],
    summary="Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.",
    response_model_by_alias=True,
)
async def delete_model(
    model: str = Path(..., description="The model to delete"),
    token_ApiKeyAuth: TokenModel = Security(get_token_ApiKeyAuth),
) -> DeleteModelResponse:
    ...


@router.get(
    "/models",
    responses={
        200: {"model": ListModelsResponse, "description": "OK"},
    },
    tags=["Models"],
    summary="Lists the currently available models, and provides basic information about each one such as the owner and availability.",
    response_model_by_alias=True,
)
async def list_models(
    token_ApiKeyAuth: TokenModel = Security(get_token_ApiKeyAuth),
) -> ListModelsResponse:
    ...


@router.get(
    "/models/{model}",
    responses={
        200: {"model": Model, "description": "OK"},
    },
    tags=["Models"],
    summary="Retrieves a model instance, providing basic information about the model such as the owner and permissioning.",
    response_model_by_alias=True,
)
async def retrieve_model(
    model: str = Path(..., description="The ID of the model to use for this request"),
    token_ApiKeyAuth: TokenModel = Security(get_token_ApiKeyAuth),
) -> Model:
    ...
