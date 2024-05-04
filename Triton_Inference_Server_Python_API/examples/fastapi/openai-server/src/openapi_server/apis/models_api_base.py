# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.delete_model_response import DeleteModelResponse
from openapi_server.models.list_models_response import ListModelsResponse
from openapi_server.models.model import Model
from openapi_server.security_api import get_token_ApiKeyAuth


class BaseModelsApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseModelsApi.subclasses = BaseModelsApi.subclasses + (cls,)

    def delete_model(
        self,
        model: str,
    ) -> DeleteModelResponse:
        ...

    def list_models(
        self,
    ) -> ListModelsResponse:
        ...

    def retrieve_model(
        self,
        model: str,
    ) -> Model:
        ...
