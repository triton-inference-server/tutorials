# generated by fastapi-codegen:
#   filename:  openai_trimmed.yml
#   timestamp: 2024-05-05T21:52:36+00:00

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from openai_protocol_types import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    DeleteModelResponse,
    ListModelsResponse,
    Model,
    ObjectType,
)

owned_by = "ACME"

model_map = {
    "llama-3-8b-instruct": Model(
        id="llama-3-8b-instruct",
        created=int(time.time()),
        object=ObjectType.model,
        owned_by=owned_by,
    )
}

import tritonserver

server = tritonserver.Server(
    model_repository="/workspace/llm-models", log_verbose=6, strict_model_config=False
).start(wait_until_ready=True)

app = FastAPI(
    title="OpenAI API",
    description="The OpenAI REST API. Please see https://platform.openai.com/docs/api-reference for more details.",
    version="2.0.0",
    termsOfService="https://openai.com/policies/terms-of-use",
    contact={"name": "OpenAI Support", "url": "https://help.openai.com/"},
    license={
        "name": "MIT",
        "url": "https://github.com/openai/openai-openapi/blob/master/LICENSE",
    },
    servers=[{"url": "https://api.openai.com/v1"}],
)


@app.post(
    "/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
def create_chat_completion(
    body: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse:
    """
    Creates a model response for the given chat conversation.
    """
    pass


@app.post("/completions", response_model=CreateCompletionResponse, tags=["Completions"])
def create_completion(body: CreateCompletionRequest) -> CreateCompletionResponse:
    """
    Creates a completion for the provided prompt and parameters.
    """
    pass


@app.get("/models", response_model=ListModelsResponse, tags=["Models"])
def list_models() -> ListModelsResponse:
    """
    Lists the currently available models, and provides basic information about each one such as the owner and availability.
    """
    return ListModelsResponse(object=ObjectType.list, data=list(model_map.values()))


@app.get("/models/{model}", response_model=Model, tags=["Models"])
def retrieve_model(model: str) -> Model:
    """
    Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    """

    if model in model_map:
        return model_map[model]

    raise HTTPException(status_code=404, detail=f"Unknown model: {model}")


@app.delete("/models/{model}", response_model=DeleteModelResponse, tags=["Models"])
def delete_model(model: str) -> DeleteModelResponse:
    """
    Delete a fine-tuned model. You must have the Owner role in your organization to delete a model.
    """
    pass
