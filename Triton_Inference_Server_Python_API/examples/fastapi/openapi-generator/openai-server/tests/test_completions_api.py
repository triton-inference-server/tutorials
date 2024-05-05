# coding: utf-8

from fastapi.testclient import TestClient
from openapi_server.models.create_completion_request import (  # noqa: F401
    CreateCompletionRequest,
)
from openapi_server.models.create_completion_response import (  # noqa: F401
    CreateCompletionResponse,
)


def test_create_completion(client: TestClient):
    """Test case for create_completion

    Creates a completion for the provided prompt and parameters.
    """
    create_completion_request = {
        "logit_bias": {"key": 1},
        "seed": -2147483648,
        "max_tokens": 16,
        "presence_penalty": 0.25495066265333133,
        "echo": 0,
        "suffix": "test.",
        "n": 1,
        "logprobs": 2,
        "top_p": 1,
        "frequency_penalty": 0.4109824732281613,
        "best_of": 1,
        "stop": "\n",
        "stream": 0,
        "temperature": 1,
        "model": "CreateCompletionRequest_model",
        "prompt": "This is a test.",
        "user": "user-1234",
    }

    headers = {
        "Authorization": "Bearer special-key",
    }
    # uncomment below to make a request
    # response = client.request(
    #    "POST",
    #    "/completions",
    #    headers=headers,
    #    json=create_completion_request,
    # )

    # uncomment below to assert the status code of the HTTP response
    # assert response.status_code == 200
