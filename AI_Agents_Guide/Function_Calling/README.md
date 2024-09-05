<!--
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
-->

# Function Calling with Triton Inference Server

This tutorial focuses on function calling, a common approach to easily connect
large language models (LLMs) to external tools. This method empowers AI agents
with effective tool usage and seamless interaction with external APIs,
significantly expanding their capabilities and practical applications.

## Table of Contents

- [What is Function Calling?](#what-is-function-calling)
- [Tutorial Overview](#tutorial-overview)
    + [Prerequisite: Hermes-2-Pro-Llama-3-8B](#prerequisite-hermes-2-pro-llama-3-8b)
- [Function Definitions](#function-definitions)
- [Prompt Engineering](#prompt-engineering)
- [Combining Everything Together](#combining-everything-together)
- [Further Optimizations](#further-optimizations)
    + [Enforcing Output Format](#enforcing-output-format)
    + [Parallel Tool Call](#parallel-tool-call)
- [References](#references)

## What is Function Calling?

Function calling refers to the ability of LLMs to:
 * Recognize when a specific function or tool needs to be used to answer a query
 or perform a task.
 * Generate a structured output containing the necessary arguments to call
 that function.
 * Integrate the results of the function call into its response.

Function calling is a powerful mechanism that allows LLMs to perform
more complex tasks (e.g. agent orchestration in multi-agent systems)
that require specific computations or data retrieval
beyond their inherent knowledge. By recognizing when a particular function
is needed, LLMs can dynamically extend their functionality, making them more
versatile and useful in real-world applications.

## Tutorial Overview

This tutorial demonstrates function calling using the
[Hermes-2-Pro-Llama-3-8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
model, which is pre-fine-tuned for this capability. We'll create a basic
stock reporting agent that provides up-to-date stock information and summarizes
recent company news.

### Prerequisite: Hermes-2-Pro-Llama-3-8B

Before proceeding, please make sure that you've successfully deployed
[Hermes-2-Pro-Llama-3-8B.](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)
model with Triton Inference Server and TensorRT-LLM backend
following [these steps.](../../Popular_Models_Guide/Hermes-2-Pro-Llama-3-8B/README.md)

> [!IMPORTANT]
> Make sure that the `tutorials` folder is mounted to `/tutorials`, when you
> start the docker container.

## Function Definitions

We'll define three functions for our stock reporting agent:
1. `get_current_stock_price`: Retrieves the current stock price for a given symbol.
2. `get_company_news`: Retrieves company news and press releases for a given stock symbol.
3. `final_answer`: Used as a no-op and to indicate the final response.

Each function includes its name, description, and input parameter schema:
 ```python
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
 ```
These function definitions will be passed to our model through a prompt,
enabling it to recognize and utilize them appropriately during the conversation.

For the actual implementations, please refer to [client_utils.py.](./artifacts/client_utils.py)

## Prompt Engineering

**Prompt engineering** is a crucial aspect of function calling, as it guides
the LLM in recognizing when and how to utilize specific functions.
By carefully crafting prompts, you can effectively define the LLM's role,
objectives, and the tools it can access, ensuring accurate and efficient task
execution.

For our task, we've organized a sample prompt structure, provided
in the accompanying [`system_prompt_schema.yml`](./artifacts/system_prompt_schema.yml)
file. This file meticulously outlines:

- **Role**: Defines the specific role the LLM is expected to perform.
- **Objective**: Clearly states the goal or desired outcome of the interaction.
- **Tools**: Lists the available functions or tools the LLM can use to achieve
its objective.
- **Schema**: Specifies the structure and format required for calling each tool
or function.
- **Instructions**: Provides a clear set of guidelines to ensure the LLM follows
the intended path and utilizes the tools appropriately.

By leveraging prompt engineering, you can enhance the LLM's ability
to perform complex tasks and integrate function calls seamlessly into
its responses, thereby maximizing its utility in various applications.

## Combining Everything Together

First, let's start Triton SDK container:
```bash
# Using the SDK container as an example
docker run --rm -it --net host --shm-size=2g \
    --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    -v /path/to/tutorials/:/tutorials \
    -v /path/to/tutorials/repo:/tutorials \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

The provided client script uses `pydantic` and `yfinance` libraries, which we
do not ship with the sdk container. Make sure to install it, before proceeding:

```bash
pip install pydantic yfinance
```

Run the provided [`client.py`](./artifacts/client.py) as follows:

```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "Tell me about Rivian. Include current stock price in your final response." -o 200
```

You should expect to see a response similar to:

```bash
+++++++++++++++++++++++++++++++++++++
RESPONSE: Rivian, with its current stock price of <CURRENT STOCK PRICE>, <NEWS SUMMARY>
+++++++++++++++++++++++++++++++++++++
```

To see what tools were "called" by our LLM, simply add `verbose` flag as follows:
```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "Tell me about Rivian. Include current stock price in your final response." -o 200 --verbose
```

This will show the step-by-step process of function calling, including:
- The tools being called
- The arguments passed to each tool
- The responses from each function call
- The final summarized response


```bash
[b'\n{\n  "step": "1",\n  "description": "Get the current stock price for Rivian",\n  "tool": "get_current_stock_price",\n  "arguments": {\n    "symbol": "RIVN"\n  }\n}']
=====================================
Executing function: get_current_stock_price({'symbol': 'RIVN'})
Function response: <CURRENT STOCK PRICE>
=====================================
[b'\n{\n  "step": "2",\n  "description": "Get company news and press releases for Rivian",\n  "tool": "get_company_news",\n  "arguments": {\n    "symbol": "RIVN"\n  }\n}']
=====================================
Executing function: get_company_news({'symbol': 'RIVN'})
Function response: [<LIST OF RECENT NEWS TITLES>]
=====================================
[b'\n{\n  "step": "3",\n  "description": "Summarize the company news and press releases for Rivian",\n  "tool": "final_answer",\n  "arguments": {\n    "final_response": "Rivian, with its current stock price of  <CURRENT STOCK PRICE>, <NEWS SUMMARY>"\n  }\n}']


+++++++++++++++++++++++++++++++++++++
RESPONSE: Rivian, with its current stock price of  <CURRENT STOCK PRICE>, <NEWS SUMMARY>
+++++++++++++++++++++++++++++++++++++
```

> [!TIP]
> In this tutorial, all functionalities (tool definitions, implementations,
> and executions) are implemented on the client side (see
> [client.py](./artifacts/client.py)).
> For production scenarios, especially when functions are known beforehand,
> consider implementing this logic on the server side.
> A recommended approach for server-side implementation is to deploy your
> workflow through a Triton [ensemble](https://github.com/triton-inference-server/server/blob/a6fff975a214ff00221790dd0a5521fb05ce3ac9/docs/user_guide/architecture.md#ensemble-models)
> or a [BLS](https://github.com/triton-inference-server/python_backend?tab=readme-ov-file#business-logic-scripting).
> Use a pre-processing model to combine and format the user prompt with the
> system prompt and available tools. Employ a post-processing model to manage
> multiple calls to the deployed LLM as needed to reach the final answer.

## Further Optimizations

### Enforcing Output Format

In this tutorial, we demonstrated how to enforce a specific output format
using prompt engineering. The desired structure is as follows:
```python
  {
    "step" : <Step number>
    "description": <Description of what the step does and its output>
    "tool": <Tool to use>,
    "arguments": {
        <Parameters to pass to the tool as a valid dict>
    }
  }
```
However, there may be instances where the output deviates from this
required schema. For example, consider the following prompt execution:

```bash
python3 /tutorials/AI_Agents_Guide/Function_Calling/artifacts/client.py --prompt "How Rivian is doing?" -o 500 --verbose
```
This execution may fail with an invalid JSON format error. The verbose
output will reveal that the final LLM response contained plain text
instead of the expected JSON format:
```
{
  "step": "3",
  "description": <Description of what the step does and its output>
  "tool": "final_answer",
  "arguments": {
    "final_response": <Final Response>
  }
}
```
Fortunately, this behavior can be controlled using constrained decoding,
a technique that guides the model to generate outputs that meet specific
formatting and content requirements.  We strongly recommend exploring our
dedicated [tutorial](../Constrained_Decoding/README.md) on constrained decoding
to gain deeper insights and enhance your ability to manage model outputs
effectively.

> [!TIP]
> For optimal results, utilize the `FunctionCall` class defined in
> [client_utils.py](./artifacts/client_utils.py) as the JSON schema
> for your Logits Post-Processor. This approach ensures consistent
> and properly formatted outputs, aligning with the structure we've
> established throughout this tutorial.

### Parallel Tool Call

This tutorial focuses on a single turn forced call, the LLM is prompted
to make a specific function call within a single interaction. This approach is
useful when a precise action is needed immediately, ensuring that
the function is executed as part of the current conversation.

It is possible, that come of function calls can be executed simultaneously.
This technique is beneficial for tasks that can be divided into independent
operations, allowing for increased efficiency and reduced response time.

We encourage our readers to take on the challenge of implementing
parallel tool calls as a practical exercise.

## References

Parts of this tutorial are based of [Hermes-Function-Calling](https://github.com/NousResearch/Hermes-Function-Calling).