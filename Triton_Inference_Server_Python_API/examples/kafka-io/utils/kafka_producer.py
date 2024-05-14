import json
from collections import deque

import numpy as np
from confluent_kafka.serialization import StringSerializer
from gcn_kafka import Producer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class KafkaProducer:
    def __init__(self, config: dict, topic: str, message_queue: deque):
        self.config = config
        self.topics = topic
        self.message_queue = message_queue
        self.serializer = StringSerializer("utf_8")

    def send_data(self):
        producer = Producer(self.config)
        self._produce(producer)

    def _produce(self, producer):
        def delivery_report(err, msg):
            """
            Reports the failure or success of a message delivery.
            Args:
                 err (KafkaError): The error that occurred on None on success.
                msg (Message): The message that was produced or failed.
            """
            if err is not None:
                print(f"Delivery failed for User record {msg.key()}: {err}")
                return
            print(
                f"User record successfully produced to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}"
            )

        while True:
            producer.poll(0.0)
            try:
                if self.message_queue.__len__() > 0:
                    message = self.message_queue.pop()
                    out = dict()
                    for output, value in message.outputs.items():
                        out[output] = np.from_dlpack(value)
                    json_message = message.__dict__
                    json_message["outputs"] = out
                    json_message["model"] = json_message["model"].__dict__
                    json_message["model"].pop("_server", None)
                    del message

                    producer.produce(
                        topic=self.topics,
                        value=self.serializer(
                            json.dumps(json_message, cls=NumpyEncoder)
                        ),
                        on_delivery=delivery_report,
                    )
                    producer.flush()
            except KeyboardInterrupt as e:
                print(f"Keyboard Interrupt received {e}")
                break
            except Exception as e:
                print(f"Error while producing the message {e}")
            finally:
                producer.flush()
        producer.close()
