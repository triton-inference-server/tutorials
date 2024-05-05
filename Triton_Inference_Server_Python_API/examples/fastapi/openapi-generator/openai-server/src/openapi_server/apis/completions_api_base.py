# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_completion_request import CreateCompletionRequest
from openapi_server.models.create_completion_response import CreateCompletionResponse

# from openapi_server.security_api import get_token_ApiKeyAuth


class BaseCompletionsApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseCompletionsApi.subclasses = BaseCompletionsApi.subclasses + (cls,)

    def create_completion(
        self,
        create_completion_request: CreateCompletionRequest,
    ) -> CreateCompletionResponse:
        ...
