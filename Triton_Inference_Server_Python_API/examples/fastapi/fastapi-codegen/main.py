# generated by fastapi-codegen:
#   filename:  openai_trimmed.yml
#   timestamp: 2024-05-05T21:52:36+00:00

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TypedDict

import numpy
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from huggingface_hub.utils import chunk_iterable
from openai_protocol_types import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    ListModelsResponse,
    Model,
    ObjectType,
)
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers_utils.tokenizer import get_tokenizer

owned_by = "ACME"
default_role = "assistant"
add_generation_prompt_default = True

model_map = {
    "llama-3-8b-instruct": Model(
        id="llama-3-8b-instruct",
        created=int(time.time()),
        object=ObjectType.model,
        owned_by=owned_by,
    )
}
tokenizer_map = {
    "llama-3-8b-instruct": get_tokenizer(
        tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct"
    )
}


import tritonserver

server = tritonserver.Server(
    model_repository="/workspace/llm-models",
    log_verbose=6,
    strict_model_config=False,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
).start(wait_until_ready=True)

for model in model_map.values():
    server.load(model.id)

server.load("preprocessing")
server.load("postprocessing")
server.load("tensorrt_llm")


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


def streaming_chat_completion_response(request_id, created, model, role, responses):
    # first chunk

    choice = ChatCompletionStreamingResponseChoice(
        index=0,
        delta=ChatCompletionStreamResponseDelta(
            role=role, content=None, function_call=None
        ),
        logprobs=None,
        finish_reason=None,
    )
    chunk = CreateChatCompletionStreamResponse(
        id=request_id,
        choices=[choice],
        created=created,
        model=model,
        system_fingerprint=None,
        object=ObjectType.chat_completion_chunk,
    )
    yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    for response in responses:
        try:
            text = response.outputs["text_output"].to_string_array()[0]
        except:
            text = str(response.outputs["text_output"].to_bytes_array()[0])

        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            ),
            logprobs=None,
            finish_reason=ChatCompletionFinishReason.stop if response.final else None,
        )

        chunk = CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
        )

        yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    yield "data: [DONE]\n\n"


@app.post(
    "/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
def create_chat_completion(
    request: CreateChatCompletionRequest,
) -> CreateChatCompletionResponse | StreamingResponse:
    """
    Creates a model response for the given chat conversation.
    """

    if request.model not in model_map:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    if request.n and request.n > 1:
        raise HTTPException(status_code=400, detail=f"Only single choice is supported")

    model = server.model(request.model)
    tokenizer = tokenizer_map[request.model]

    conversation = [
        {"role": str(message.role), "content": str(message.content)}
        for message in request.messages
    ]

    prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt_default,
    )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())
    exclude_input_in_output = True

    sampling_parameters = request.copy(exclude={"model", "stream", "messages"}).dict()

    responses = model.infer(
        inputs={
            "text_input": [[prompt]],
            "stream": [[request.stream]],
            "max_tokens": [[numpy.int32(request.max_tokens)]]
            #            "exclude_input_in_output": [exclude_input_in_output],
        },
        #        parameters=sampling_parameters,
    )

    if request.stream:
        return StreamingResponse(
            streaming_chat_completion_response(
                request_id, created, request.model, conversation[-1]["role"], responses
            )
        )

    response = list(responses)[0]

    try:
        print(response)
        print(response.outputs)
        text = response.outputs["text_output"].to_string_array()[0]
    except:
        text = str(response.outputs["text_output"].to_bytes_array()[0])

    return CreateChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=text, role=default_role, function_call=None
                ),
                logprobs=None,
                finish_reason=ChatCompletionFinishReason.stop,
            )
        ],
        created=created,
        model=request.model,
        system_fingerprint=None,
        object=ObjectType.chat_completion,
    )


def streaming_completion_response(request_id, created, model, responses):
    for response in responses:
        try:
            text = response.outputs["text_output"].to_string_array()[0]
        except:
            text = str(response.outputs["text_output"].to_bytes_array()[0])

        choice = Choice(
            finish_reason=FinishReason.stop if response.final else None,
            index=0,
            logprobs=None,
            text=text,
        )
        response = CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=model,
        )

        yield f"data: {response.json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/completions", response_model=CreateCompletionResponse, tags=["Completions"])
def create_completion(
    request: CreateCompletionRequest, raw_request: Request
) -> CreateCompletionResponse | StreamingResponse:
    """
    Creates a completion for the provided prompt and parameters.
    """
    if request.suffix is not None:
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    if request.model not in model_map:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
    else:
        model = server.model(model_map[request.model].id)

    if request.prompt is None:
        request.prompt = "<|endoftext|>"

    # Currently only support single string as input
    if not isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400, detail="only single string input is supported"
        )

    if request.logit_bias is not None or request.logprobs is not None:
        raise HTTPException(
            status_code=400, detail="logit bias and log probs not supported"
        )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())
    exclude_input_in_output = True

    if request.echo:
        exclude_input_in_output = False

    sampling_parameters = request.copy(
        exclude={"model", "prompt", "stream", "echo"}
    ).dict()

    responses = model.infer(
        inputs={
            "text_input": [request.prompt],
            "stream": [request.stream],
            "exclude_input_in_output": [exclude_input_in_output],
        },
        parameters=sampling_parameters,
    )
    if request.stream:
        return StreamingResponse(
            streaming_completion_response(request_id, created, request.model, responses)
        )
    response = list(responses)[0]

    try:
        text = response.outputs["text_output"].to_string_array()[0]
    except:
        text = str(response.outputs["text_output"].to_bytes_array()[0])

    choice = Choice(
        finish_reason=FinishReason.stop if response.final else None,
        index=0,
        logprobs=None,
        text=text,
    )
    return CreateCompletionResponse(
        id=request_id,
        choices=[choice],
        system_fingerprint=None,
        object=ObjectType.text_completion,
        created=created,
        model=request.model,
    )


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
