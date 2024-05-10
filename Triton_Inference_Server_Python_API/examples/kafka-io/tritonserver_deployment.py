import json
import threading
from collections import deque
from pprint import pprint

import certifi
import click
import numpy as np
import tritonserver

from utils.kafka_consumer import KafkaConsumer
from utils.kafka_producer import KafkaProducer


def _print_heading(message):
    print("")
    print(message)
    print("-" * len(message))


def check_ssl_requirement(config: dict) -> dict:
    if "SSL" in config.get("security.protocol", None):
        if "ssl.ca.location" not in config:
            config["ssl.ca.location"] = certifi.where()
    return config


class TritonKafkaEndpoint:
    def __init__(self, consumer_queue: deque, producer_queue: deque, model_repository: str):
        self.producer_queue = producer_queue
        self.queue = consumer_queue
        self._triton_server = tritonserver.Server(model_repository=model_repository,
                                                  model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
                                                  log_info=True)
        self._triton_server.start(wait_until_ready=True)
        _print_heading("Triton Server Started")
        _print_heading("Metadata")
        pprint(self._triton_server.metadata())

    def infer(self):
        if not self._triton_server.model("tokenizer").ready():
            try:
                self._tokenizer_model = self._triton_server.load(
                    "tokenizer"
                )

                if not self._tokenizer_model.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                print("Error can't load tokenizer model!")
                print(
                    f"Please ensure dependencies are met and you have set the environment variables if any {error}"
                )
                return
        _print_heading("Models")
        pprint(self._triton_server.models())

        while True:
            if self.queue.__len__() > 0:
                message = self.queue.pop()
                try:
                    response = self._triton_server.model('tokenizer').infer(inputs={'TEXT': np.array([message])})
                    for resp in response:
                        self.producer_queue.append(resp)
                except Exception as e:
                    print(e)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--consumer_config", type=str, required=True)
@click.option("--producer_config", type=str, required=True)
@click.option("--consumer_topics", type=str, multiple=True)
@click.option("--producer_topic", type=str, default="data-tokenizer-output")
@click.option("--model_repository", type=str, default="/models")
def main(consumer_config: str, producer_config: str, consumer_topics: list, producer_topic: str, model_repository: str):
    consumer_config = check_ssl_requirement(json.loads(consumer_config))
    producer_config = check_ssl_requirement(json.loads(producer_config))
    consumer_queue = deque()
    producer_queue = deque()
    consumer = KafkaConsumer(consumer_config, list(consumer_topics), consumer_queue)
    producer = KafkaProducer(producer_config, producer_topic, producer_queue)
    tse = TritonKafkaEndpoint(consumer_queue, producer_queue, model_repository)
    threading.Thread(target=tse.infer, daemon=False).start()
    print("Starting Producer")
    threading.Thread(target=producer.send_data, daemon=False).start()
    print("Starting Consumer")
    threading.Thread(target=consumer.read, daemon=False).start()


if __name__ == "__main__":
    print("starting TritonKafkaEndpoint")
    main()
