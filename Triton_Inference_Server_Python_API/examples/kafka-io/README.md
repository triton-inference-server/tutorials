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

# Triton Inference Server Kafka I/O Deployment

Using the Triton Inference Server In-Process Python API you can
integrate triton server based models into any Python framework
to consume the messages from a Kafka topic and produce the inference results
back to the kafka topic of choice.

This directory contains an example Triton Inference Server
deployment based on Kafka I/O that uses threads for each of server, consumer and producer.

| [Installation](#installation) | [Run Deployment](#starting-the-pipeline) | [Send Requests](#send-requests-to-deployment) |


## Installation

In this Kafka I/O pipeline we deploy a pre-processing stage of tokenization based on `transformers` tokenization module and can be extended to any type of models as needed.

### Pre-requisite
1. [Docker](https://docs.docker.com/engine/install/)

### Starting docker container
Once you have the docker service up and running, launch a container by executing the following command:

```bash
docker run --rm -it --gpus all -v <path>/<to>/tutorials/Triton_Inference_Server_Python_API/examples/kafka-io/:/opt/tritonserver/kafka-io -w /opt/tritonserver/kafka-io  --entrypoint bash nvcr.io/nvidia/tritonserver:24.06-py3
```

### Clone Repository

```bash
git clone https://github.com/triton-inference-server/tutorials.git
cd tutorials/Triton_Inference_Server_Python_API/examples/kafka-io
```

*Note: Skip this step if you have mounted the git repository from local directory to the docker container*


### Install dependencies

Please note that installation times may vary depending on
your hardware configuration and network connection.


```bash
pip install -r requirements.txt
```

If triton server is not already installed, install the dependency by using the following command.

```bash
pip install /opt/tritonserver/python/tritonserver-2.44.0-py3-none-any.whl
```

Next run the provided `start-kafka.sh` script that will perform the following actions:
1. Download kafka and it's dependencies
2. Start Kafka service by starting Zookeeper and Kafka brokers
3. Create 2 new topics with names `inference-input` as input queue and `inference-output` to store the inference results

```bash
chmod +x start-kafka.sh
./start-kafka.sh
```

## Starting the pipeline

### Start the inference pipeline

Run the provided `start-server.sh` script that will perform the following actions:
1. Export Kafka Producer and Consumer configs, topic names for input and output topics, model name and repositories.
2. Start the server.

```bash
chmod +x start-server.sh
./start-server.sh
```

When your console outputs something similar to:
```bash
2024-07-18 21:55:38,254 INFO api.py:609 -- Deployed app 'default' successfully.
```
It means that the server has started successfully. You can press `Ctrl+C` and proceed to the next steps.

*Note: In the above invocation, we are using default of 1 thread for kafka consumer, however, if you need to increase the concurrency, please set the environment variable `KAFKA_CONSUMER_MAX_WORKER_THREADS` to the desired value and restart the server. This should start the server with new concurrency of the consumer to increase the throughput of the deployment*

## Send Requests to Deployment

In order to send requests to inference pipeline deployed, produce messages into the input kafka topic using the following command.

```bash
cd kafka_2.13-3.7.0
bin/kafka-console-producer.sh --topic inference-input --bootstrap-server localhost:9092
```

Once, the above command has been executed, you should see a prompt `>` to start ingesting the messages to the input topic.

```bash
> this is a sample message
>
```

Once you have produced enough messages, you can exit the prompt by pressing `Ctrl+C`.

#### Example Output
Once the workflow consumes the ingested messages from the kafka topic, it invokes the triton server and produces the inference output as `json` string to the output kafka topic. Once the message has been ingested, we can start the consumer to see the output messages from the pipeline ingested to the output topic

```bash
bin/kafka-console-consumer.sh --topic inference-output --from-beginning --bootstrap-server localhost:9092
```

Since, our example has a tokenizer deployed as a custom model in triton, we should see an output inserted into kafka topic as shown below.

```bash
{"model": {"name": "tokenizer", "version": 1, "state": null, "reason": null}, "request_id": "", "parameters": {}, "outputs": {"input_ids": [[101, 1142, 1110, 2774, 3802, 118, 1207, 130, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "token_type_ids": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}, "error": null, "classification_label": null, "final": true}
```
