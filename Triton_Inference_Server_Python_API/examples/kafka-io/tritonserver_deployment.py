import json
import os
import threading
from collections import deque
from pprint import pprint
from typing import List

import certifi
import numpy as np
import tritonserver
from fastapi import FastAPI, Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from tritonserver._c.triton_bindings import TRITONSERVER_DataType
from utils.kafka_consumer import KafkaConsumer
from utils.kafka_producer import KafkaProducer, NumpyEncoder

app = FastAPI()


def check_ssl_requirement(config: dict) -> dict:
    print(type(config))
    if "SSL" in config.get("security.protocol"):
        if "ssl.ca.location" not in config:
            config["ssl.ca.location"] = certifi.where()
    return config


def get_consumer_configs() -> dict:
    configs = dict()
    configs["bootstrap.servers"] = os.environ.get("CONSUMER_KAFKA_SERVER", None)
    configs["sasl.mechanisms"] = os.environ.get("CONSUMER_SASL_MECHANISM", None)
    configs["sasl.oauthbearer.method"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_METHOD", None
    )
    configs["sasl.oauthbearer.scope"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_SCOPE", None
    )
    configs["sasl.oauthbearer.client.id"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_CLIENT_ID", None
    )
    configs["sasl.oauthbearer.client.secret"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_CLIENT_SECRET", None
    )
    configs["sasl.oauthbearer.token.endpoint.url"] = os.environ.get(
        "CONSUMER_SASL_OAUTHBEARER_TOKEN_ENDPOINT", None
    )
    configs["security.protocol"] = os.environ.get("CONSUMER_SECURITY_PROTOCOL", None)
    return configs


def get_producer_configs() -> dict:
    configs = dict()
    configs["bootstrap.servers"] = os.environ.get("PRODUCER_KAFKA_SERVER", None)
    configs["sasl.mechanisms"] = os.environ.get("PRODUCER_SASL_MECHANISM", None)
    configs["sasl.oauthbearer.method"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_METHOD", None
    )
    configs["sasl.oauthbearer.scope"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_SCOPE", None
    )
    configs["sasl.oauthbearer.client.id"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_CLIENT_ID", None
    )
    configs["sasl.oauthbearer.client.secret"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_CLIENT_SECRET", None
    )
    configs["sasl.oauthbearer.token.endpoint.url"] = os.environ.get(
        "PRODUCER_SASL_OAUTHBEARER_TOKEN_ENDPOINT", None
    )
    configs["security.protocol"] = os.environ.get("PRODUCER_SECURITY_PROTOCOL", None)
    return configs


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, distilbert_model_handle: DeploymentHandle) -> None:
        self.handle = distilbert_model_handle
        producer_queue = deque()
        consumer_config = json.loads(os.environ.get("CONSUMER_CONFIGS"))
        consumer_config = check_ssl_requirement(consumer_config)
        consumer = KafkaConsumer(
            consumer_config,
            (os.environ.get("CONSUMER_TOPICS", "")).split(","),
            self.handle,
            producer_queue,
        )
        producer_config = json.loads(os.environ.get("PRODUCER_CONFIGS"))
        producer_config = check_ssl_requirement(producer_config)
        producer = KafkaProducer(
            producer_config, os.environ.get("PRODUCER_TOPIC", ""), producer_queue
        )
        print("Starting Producer")
        threading.Thread(
            target=producer.send_data, daemon=False
        ).start()  # convert to thread pool
        print("Starting Consumer")
        threading.Thread(target=consumer.read, daemon=False).start()

    @app.post("/infer")
    async def classify(self, request: Request):  # sentence: str):
        data = await request.json()
        print(data)
        return await self.handle.infer.remote(data.get("sentences"))

    @app.get("/health")
    async def health(self):
        return "OK"


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class TritonKafkaEndpoint:
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME")
        self.model_input_name = os.environ.get("MODEL_INPUT_NAME")
        self.model_repository = os.environ.get("MODEL_REPOSITORY")
        self._triton_server = tritonserver.Server(
            model_repository=self.model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=True,
            metrics=True,
            gpu_metrics=True,
            cpu_metrics=True,
        )
        self._triton_server.start(wait_until_ready=True)
        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())
        if not self._triton_server.model(self.model_name).ready():
            try:
                self._tokenizer_model = self._triton_server.load(self.model_name)

                if not self._tokenizer_model.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load tokenizer model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variables if any {error}"
                )

    def infer(self, message: List[str]):
        responses = self._triton_server.model(self.model_name).infer(
            inputs={self.model_input_name: np.array(message)}
        )
        result = list()
        for response in responses:
            out = dict()
            for output, value in response.outputs.items():
                if value.data_type == TRITONSERVER_DataType.BYTES:
                    out[output] = value.to_string_array()
                else:
                    out[output] = np.from_dlpack(value)
            json_message = response.__dict__
            json_message["outputs"] = out
            json_message["model"] = json_message["model"].__dict__
            json_message["model"].pop("_server", None)
            result.append(json.dumps(json_message, cls=NumpyEncoder))
        return {"result": result}


triton_handle = TritonKafkaEndpoint.bind()
entrypoint = APIIngress.bind(triton_handle)
