import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from typing import List

from confluent_kafka import KafkaError, KafkaException
from gcn_kafka import Consumer
from ray.serve.handle import DeploymentHandle


class KafkaConsumer:
    def __init__(
        self,
        config: dict,
        topics: List[str],
        triton_server_handle: DeploymentHandle,
        output_queue: deque,
    ):
        self.config = config
        self.topics = topics
        self.triton_handle = triton_server_handle
        self.output_queue = output_queue

    def read(self):
        consumer = Consumer(self.config)
        consumer.subscribe(self.topics)
        self._consume_data(consumer)

    def _infer(self, future):
        print("The custom callback was called.")
        result = future.result()
        self.output_queue.append(result.result())
        print(f"Got: {future.result()}")

    def _consume_data(self, consumer):
        while True:
            try:
                msg = consumer.poll(0.1)
                if not msg:
                    continue
                if msg.error():
                    print(msg.error())
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(
                            f"End of partition has been reached {msg.topic()}/{msg.partition()}"
                        )
                    else:
                        raise KafkaException(msg.error())
                print(f"Key: {msg.key()}, Value: {msg.value()}")
                with ThreadPoolExecutor(
                    max_workers=int(
                        os.environ.get("KAFKA_CONSUMER_MAX_WORKER_THREADS", 1)
                    )
                ) as executor:
                    future = executor.submit(
                        self.triton_handle.infer.remote, [msg.value()]
                    )
                    future.add_done_callback(self._infer)
            except KeyboardInterrupt as e:
                print(f"Keyboard Interrupt Received: {e}")
                break
            except Exception as e:
                print(f"Exception {e}")
        consumer.close()
