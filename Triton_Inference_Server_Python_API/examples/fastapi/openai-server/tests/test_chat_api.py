# coding: utf-8

from fastapi.testclient import TestClient

from openapi_server.models.create_chat_completion_request import (  # noqa: F401
    CreateChatCompletionRequest,
)
from openapi_server.models.create_chat_completion_response import (  # noqa: F401
    CreateChatCompletionResponse,
)


def test_create_chat_completion(client: TestClient):
    """Test case for create_chat_completion

    Creates a model response for the given chat conversation.
    """
    create_chat_completion_request = {
        "top_logprobs": 2,
        "logit_bias": {"key": 6},
        "seed": -2147483648,
        "functions": [
            {"name": "name", "description": "description", "parameters": {"key": ""}},
            {"name": "name", "description": "description", "parameters": {"key": ""}},
            {"name": "name", "description": "description", "parameters": {"key": ""}},
            {"name": "name", "description": "description", "parameters": {"key": ""}},
            {"name": "name", "description": "description", "parameters": {"key": ""}},
        ],
        "max_tokens": 5,
        "function_call": "none",
        "presence_penalty": 0.25495066265333133,
        "tools": [
            {
                "function": {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                "type": "function",
            },
            {
                "function": {
                    "name": "name",
                    "description": "description",
                    "parameters": {"key": ""},
                },
                "type": "function",
            },
        ],
        "n": 1,
        "logprobs": 0,
        "top_p": 1,
        "frequency_penalty": -1.6796687238155954,
        "response_format": {"type": "json_object"},
        "stop": "CreateChatCompletionRequest_stop",
        "stream": 0,
        "temperature": 1,
        "messages": [
            {"role": "system", "name": "name", "content": "content"},
            {"role": "system", "name": "name", "content": "content"},
        ],
        "tool_choice": "none",
        "model": "gpt-4-turbo",
        "user": "user-1234",
    }

    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    # response = client.request(
    #    "POST",
    #    "/chat/completions",
    #    headers=headers,
    #    json=create_chat_completion_request,
    # )

    # uncomment below to assert the status code of the HTTP response
    # assert response.status_code == 200
