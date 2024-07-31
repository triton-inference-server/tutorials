#!/usr/bin/python

import argparse
import json
import re
import sys

import end_to_end_grpc_client as client_utils
import numpy as np
import tritonclient.grpc as grpcclient
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="Inference server URL."
    )

    parser.add_argument("-p", "--prompt", type=str, required=True, help="Input prompt.")

    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="ensemble",
        choices=["ensemble", "tensorrt_llm_bls"],
        help="Name of the Triton model to send request to",
    )

    parser.add_argument(
        "-S",
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument(
        "-o",
        "--output-len",
        type=int,
        default=100,
        required=False,
        help="Specify output length",
    )

    parser.add_argument(
        "--request-id",
        type=str,
        default="",
        required=False,
        help="The request_id for the stop request",
    )

    parser.add_argument("--stop-words", nargs="+", default=[], help="The stop words")

    parser.add_argument("--bad-words", nargs="+", default=[], help="The bad words")

    parser.add_argument(
        "--embedding-bias-words", nargs="+", default=[], help="The biased words"
    )

    parser.add_argument(
        "--embedding-bias-weights",
        nargs="+",
        default=[],
        help="The biased words weights",
    )

    parser.add_argument(
        "--overwrite-output-text",
        action="store_true",
        required=False,
        default=False,
        help="In streaming mode, overwrite previously received output text instead of appending to it",
    )

    parser.add_argument(
        "--return-context-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return context logits, the engine must be built with gather_context_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--return-generation-logits",
        action="store_true",
        required=False,
        default=False,
        help="Return generation logits, the engine must be built with gather_ generation_logits or gather_all_token_logits",
    )

    parser.add_argument(
        "--end-id", type=int, required=False, help="The token id for end token."
    )

    parser.add_argument(
        "--pad-id", type=int, required=False, help="The token id for pad token."
    )

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8001"

    embedding_bias_words = (
        FLAGS.embedding_bias_words if FLAGS.embedding_bias_words else None
    )
    embedding_bias_weights = (
        FLAGS.embedding_bias_weights if FLAGS.embedding_bias_weights else None
    )

    try:
        client = grpcclient.InferenceServerClient(url=FLAGS.url)
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    return_context_logits_data = None
    if FLAGS.return_context_logits:
        return_context_logits_data = np.array(
            [[FLAGS.return_context_logits]], dtype=bool
        )

    return_generation_logits_data = None
    if FLAGS.return_generation_logits:
        return_generation_logits_data = np.array(
            [[FLAGS.return_generation_logits]], dtype=bool
        )

    prompt = utils.process_prompt(FLAGS.prompt)

    functions = utils.MyFunctions()

    while True:
        output_text = client_utils.run_inference(
            client,
            prompt,
            FLAGS.output_len,
            FLAGS.request_id,
            FLAGS.repetition_penalty,
            FLAGS.presence_penalty,
            FLAGS.frequency_penalty,
            FLAGS.temperature,
            FLAGS.stop_words,
            FLAGS.bad_words,
            embedding_bias_words,
            embedding_bias_weights,
            FLAGS.model_name,
            FLAGS.streaming,
            FLAGS.beam_width,
            FLAGS.overwrite_output_text,
            return_context_logits_data,
            return_generation_logits_data,
            FLAGS.end_id,
            FLAGS.pad_id,
            FLAGS.verbose,
        )

        print("\n\n+++++++++++++++++++++++++++++++++++++")
        print(f"RESPONSE: {output_text}")
        print("+++++++++++++++++++++++++++++++++++++\n\n")

        if "<tool_call>" not in output_text:
            break

        functions_to_call = json.loads(
            re.findall(r"<tool_call>\n(.*?)\n</tool_call>", output_text)[0]
        )
        function_name = functions_to_call["name"]
        function_args = functions_to_call["arguments"]
        function_to_call = getattr(functions, function_name)
        print("=====================================")
        print(f"Executing function: {function_name}({function_args}) ")
        print("=====================================")
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'

        prompt += f"{output_text}<|im_end|><|im_start|>tool\n<tool_response>\n{results_dict}\n</tool_response><|im_end|><|im_start|>assistant"
