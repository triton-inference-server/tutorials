# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
import yfinance as yf
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype

###############################################################################
# TOOLS Definition and Implementation                                         #
###############################################################################

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_stock_price",
            "description": "Get the current stock price for a given symbol.\n\nArgs:\n  symbol (str): The stock symbol.\n\nReturns:\n  float: The current stock price, or None if an error occurs.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get company news and press releases for a given stock symbol.\n\nArgs:\nsymbol (str): The stock symbol.\n\nReturns:\npd.DataFrame: DataFrame containing company news and press releases.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return final generated answer",
            "parameters": {
                "type": "object",
                "properties": {"final_response": {"type": "string"}},
                "required": ["final_response"],
            },
        },
    },
]


class MyFunctions:
    def get_company_news(self, symbol: str) -> pd.DataFrame:
        """
        Get company news and press releases for a given stock symbol.

        Args:
        symbol (str): The stock symbol.

        Returns:
        pd.DataFrame: DataFrame containing company news and press releases.
        """
        try:
            news = yf.Ticker(symbol).news
            title_list = []
            for entry in news:
                title_list.append(entry["title"])
            return title_list
        except Exception as e:
            print(f"Error fetching company news for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_stock_price(self, symbol: str) -> float:
        """
        Get the current stock price for a given symbol.

        Args:
            symbol (str): The stock symbol.

        Returns:
            float: The current stock price, or None if an error occurs.
        """
        try:
            stock = yf.Ticker(symbol)
            # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
            current_price = stock.info.get(
                "regularMarketPrice", stock.info.get("currentPrice")
            )
            return current_price if current_price else None
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None


###############################################################################
# Helper Schemas                                                              #
###############################################################################


class FunctionCall(BaseModel):
    step: str
    """Step number for the action sequence"""
    description: str
    """Description of what the step does and its output"""

    tool: str
    """The name of the tool to call."""

    arguments: dict
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """


class PromptSchema(BaseModel):
    Role: str
    """Defines the specific role the LLM is expected to perform."""
    Objective: str
    """States the goal or desired outcome of the interaction."""
    Tools: str
    """A set of available functions or tools the LLM can use to achieve its
    objective."""
    Schema: str
    """ Specifies the structure and format required for calling each tool
    or function."""
    Instructions: str
    """Provides a clear set of guidelines to ensure the LLM follows
    the intended path and utilizes the tools appropriately."""


###############################################################################
# Prompt processing helper functions                                          #
###############################################################################


def read_yaml_file(file_path: str) -> PromptSchema:
    """
    Reads a YAML file and converts its content into a PromptSchema object.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        PromptSchema: An object containing the structured prompt data.
    """
    with open(file_path, "r") as file:
        yaml_content = yaml.safe_load(file)

    prompt_schema = PromptSchema(
        Role=yaml_content.get("Role", ""),
        Objective=yaml_content.get("Objective", ""),
        Tools=yaml_content.get("Tools", ""),
        Schema=yaml_content.get("Schema", ""),
        Instructions=yaml_content.get("Instructions", ""),
    )
    return prompt_schema


def format_yaml_prompt(prompt_schema: PromptSchema, variables: Dict) -> str:
    """
    Formats the prompt schema with provided variables.

    Args:
        prompt_schema (PromptSchema): The prompt schema to format.
        variables (Dict): A dictionary of variables to insert into the prompt.

    Returns:
        str: The formatted prompt string.
    """
    formatted_prompt = ""
    for field, value in prompt_schema.model_dump().items():
        formatted_value = value.format(**variables)
        if field == "Instructions":
            formatted_prompt += f"{formatted_value}"
        else:
            formatted_value = formatted_value.replace("\n", " ")
            formatted_prompt += f"{formatted_value}"
    return formatted_prompt


def process_prompt(
    user_prompt,
    system_prompt_yml=Path(__file__).parent.joinpath("./system_prompt_schema.yml"),
    tools=TOOLS,
    schema_json=FunctionCall.model_json_schema(),
):
    """
    Combines and formats the user prompt with a system prompt for model
    processing.

    This function reads a system prompt from a YAML file, formats it with the
    provided tools and schema, and integrates it with the user's original
    prompt. The result is a structured prompt ready for input into a
    language model.

    Args:
        user_prompt (str): The initial prompt provided by the user.
        system_prompt_yml (str, optional): The file path to the system prompt
            defined in a YAML file. Defaults to "./system_prompt_schema.yml".
        tools (list, optional): A list of tools available for the prompt.
            Defaults to the global TOOLS variable.
        schema_json (dict, optional): A JSON schema for a generated function call.
            Defaults to the schema from FunctionCall.model_json_schema().

    Returns:
        str: A formatted prompt string ready for use by the language model.
    """
    prompt_schema = read_yaml_file(system_prompt_yml)
    variables = {"tools": tools, "schema": schema_json}
    sys_prompt = format_yaml_prompt(prompt_schema, variables)
    processed_prompt = f"<|im_start|>system\n {sys_prompt}<|im_end|>\n"
    processed_prompt += f"<|im_start|>user\n {user_prompt}\nThis is the first turn and you don't have <tool_results> to analyze yet. <|im_end|>\n <|im_start|>assistant"
    return processed_prompt


###############################################################################
# Triton client helper functions                                              #
###############################################################################


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def run_inference(
    triton_client,
    prompt,
    output_len,
    request_id,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    temperature,
    stop_words,
    bad_words,
    embedding_bias_words,
    embedding_bias_weights,
    model_name,
    streaming,
    beam_width,
    overwrite_output_text,
    return_context_logits_data,
    return_generation_logits_data,
    end_id,
    pad_id,
    verbose,
    num_draft_tokens=0,
    use_draft_logits=None,
):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * output_len
    streaming_data = np.array([[streaming]], dtype=bool)
    beam_width_data = np.array([[beam_width]], dtype=np.int32)
    temperature_data = np.array([[temperature]], dtype=np.float32)

    inputs = [
        prepare_tensor("text_input", input0_data),
        prepare_tensor("max_tokens", output0_len),
        prepare_tensor("stream", streaming_data),
        prepare_tensor("beam_width", beam_width_data),
        prepare_tensor("temperature", temperature_data),
    ]

    if num_draft_tokens > 0:
        inputs.append(
            prepare_tensor(
                "num_draft_tokens", np.array([[num_draft_tokens]], dtype=np.int32)
            )
        )
    if use_draft_logits is not None:
        inputs.append(
            prepare_tensor(
                "use_draft_logits", np.array([[use_draft_logits]], dtype=bool)
            )
        )

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        inputs += [prepare_tensor("bad_words", bad_words_list)]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        inputs += [prepare_tensor("stop_words", stop_words_list)]

    if repetition_penalty is not None:
        repetition_penalty = [[repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
        inputs += [prepare_tensor("repetition_penalty", repetition_penalty_data)]

    if presence_penalty is not None:
        presence_penalty = [[presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)
        inputs += [prepare_tensor("presence_penalty", presence_penalty_data)]

    if frequency_penalty is not None:
        frequency_penalty = [[frequency_penalty]]
        frequency_penalty_data = np.array(frequency_penalty, dtype=np.float32)
        inputs += [prepare_tensor("frequency_penalty", frequency_penalty_data)]

    if return_context_logits_data is not None:
        inputs += [
            prepare_tensor("return_context_logits", return_context_logits_data),
        ]

    if return_generation_logits_data is not None:
        inputs += [
            prepare_tensor("return_generation_logits", return_generation_logits_data),
        ]

    if (embedding_bias_words is not None and embedding_bias_weights is None) or (
        embedding_bias_words is None and embedding_bias_weights is not None
    ):
        assert 0, "Both embedding bias words and weights must be specified"

    if embedding_bias_words is not None and embedding_bias_weights is not None:
        assert len(embedding_bias_words) == len(
            embedding_bias_weights
        ), "Embedding bias weights and words must have same length"
        embedding_bias_words_data = np.array([embedding_bias_words], dtype=object)
        embedding_bias_weights_data = np.array(
            [embedding_bias_weights], dtype=np.float32
        )
        inputs.append(prepare_tensor("embedding_bias_words", embedding_bias_words_data))
        inputs.append(
            prepare_tensor("embedding_bias_weights", embedding_bias_weights_data)
        )
    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs += [prepare_tensor("end_id", end_id_data)]

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs += [prepare_tensor("pad_id", pad_id_data)]

    user_data = UserData()
    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    # Send request
    triton_client.async_stream_infer(model_name, inputs, request_id=request_id)

    # Wait for server to close the stream
    triton_client.stop_stream()

    # Parse the responses
    output_text = ""
    while True:
        try:
            result = user_data._completed_requests.get(block=False)
        except Exception:
            break

        if type(result) == InferenceServerException:
            print("Received an error from server:")
            print(result)
        else:
            output = result.as_numpy("text_output")
            if streaming and beam_width == 1:
                new_output = output[0].decode("utf-8")
                if overwrite_output_text:
                    output_text = new_output
                else:
                    output_text += new_output
            else:
                output_text = output[0].decode("utf-8")
                if verbose:
                    print(
                        str("\n[VERBOSE MODE] LLM's response:" + output_text),
                        flush=True,
                    )

            if return_context_logits_data is not None:
                context_logits = result.as_numpy("context_logits")
                if verbose:
                    print(f"context_logits.shape: {context_logits.shape}")
                    print(f"context_logits: {context_logits}")
            if return_generation_logits_data is not None:
                generation_logits = result.as_numpy("generation_logits")
                if verbose:
                    print(f"generation_logits.shape: {generation_logits.shape}")
                    print(f"generation_logits: {generation_logits}")

    if streaming and beam_width == 1:
        if verbose:
            print(output_text)

    return output_text
