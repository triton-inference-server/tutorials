# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from openapi_server.models.create_chat_completion_request import (
    CreateChatCompletionRequest,
)
from openapi_server.models.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from openapi_server.security_api import get_token_ApiKeyAuth


class BaseChatApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseChatApi.subclasses = BaseChatApi.subclasses + (cls,)

    def create_chat_completion(
        self,
        create_chat_completion_request: CreateChatCompletionRequest,
    ) -> CreateChatCompletionResponse:
        ...
