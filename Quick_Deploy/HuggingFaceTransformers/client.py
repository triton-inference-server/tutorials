#!/usr/bin/env python3
import sys
import time
import json
import argparse
import tritonclient
from functools import partial
try:
    import tritonclient.http
    import numpy as np
    from tritonclient.utils import triton_to_np_dtype, InferenceServerException

    np.set_printoptions(threshold=10, linewidth=80, edgeitems=2)
except:
    sys.exit(
        "ERROR: tritonclient package not found, try 'pip install tritonclient[all]'"
    )

try:
    from rich import print
    from rich.table import Table
except:
    sys.exit("ERROR: rich package not found, try 'pip install rich'")


def SetupClient():
    """
    Create a simple Triton http client.
    """
    import tritonclient.http as tc

    url = "localhost:8000"
    client = tc.InferenceServerClient(url=url)
    return client, tc


def GetMetadata():
    """
    Display the specified model's metadata.
    """
    try:
        metadata = client.get_model_metadata(model_name=args.model)
        cfg_table = Table(
            "Key",
            "Value",
            show_header=True,
            title="Model Metadata",
            title_style="bold green",
        )
        for k, v in metadata.items():
            cfg_table.add_row(k, str(v))
        print(cfg_table)
    except Exception as e:
        sys.exit(f"ERROR: Triton client failed: {e}")

    return metadata


def SetupInputs():
    """
    Create an inference request from the user prompt.
    """
    inputs = []
    # Extract name, shape and data type
    input_config = metadata["inputs"][0]
    name, shape = input_config["name"], input_config["shape"]
    triton_dtype = "BYTES"
    # Ensure a prompt was specified
    if not args.prompt:
        sys.exit("ERROR: --prompt required for LLM model")
    # Create an inference request input with the collected data
    data = np.array([args.prompt], dtype=np.object_)
    inputs.append(tc.InferInput(name, shape, triton_dtype))
    data_kwargs = {"binary_data": False}
    inputs[-1].set_data_from_numpy(data, **data_kwargs)
    # Add the prompt input to the final table
    display_data = np.array_str(data)
    io_table.add_row(name, str(shape), display_data)

    return inputs


def Infer():
    """
    Send an inference request to Triton over http.
    """
    infer = partial(client.infer, args.model, inputs)
    try:
        results = []
        print("Sending inference request...")
        results.append(infer())
    except Exception as e:
        sys.exit(f"[ERROR] Inference failed: {e}")

    print("Processing inference result...")
    ProcessInferResult(results[0])
    print(io_table)


def ProcessInferResult(result):
    """
    Extract and display outputs.
    """
    response = result.get_response()
    # Process response outputs
    if response.get("outputs"):
        io_table.add_section()
        for output in response["outputs"]:
            np_data = result.as_numpy(output["name"])
            # Convert np arrays into strings
            if np_data.dtype == np.object_:
                texts = np.array([text.decode("utf-8") for text in np_data])
                texts = "\n\n".join([text for text in texts])
                texts = np.array_str(np.array(texts))
            else:
                texts = np.array_str(np_data)
            io_table.add_row(output["name"], str(np_data.shape), texts)


def ParseArgs():
    """
    Arguement parser with minimal parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="Prompt for LLM models.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # CLI Args
    args = ParseArgs()

    # Initialize client
    client, tc = SetupClient()

    # Initialize table for visualization
    io_table = Table(
        "Name",
        "Shape",
        "Data",
        show_header=True,
        title="Inputs/Outputs",
        title_style="bold green",
    )

    # Display specified model's metadata
    metadata = GetMetadata()

    # Initialize inputs to use for inference
    inputs = SetupInputs()

    # Do inference and display results
    Infer()

