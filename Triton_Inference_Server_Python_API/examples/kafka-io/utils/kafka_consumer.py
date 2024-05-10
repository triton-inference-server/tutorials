from collections import deque
from typing import List

from confluent_kafka import KafkaError, KafkaException
from gcn_kafka import Consumer


class KafkaConsumer:
    def __init__(self, config: dict, topics: List[str], message_queue: deque):
        self.config = config
        self.topics = topics
        self.message_queue = message_queue

    def read(self):
        consumer = Consumer(self.config)
        consumer.subscribe(self.topics)
        self._consume_data(consumer)

    def _consume_data(self, consumer):
        while True:
            try:
                msg = consumer.poll(0.1)
                if not msg:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(f"End of partition has been reached {msg.topic()}/{msg.partition()}")
                    else:
                        raise KafkaException(msg.error())
                print(f"Key: {msg.key()}, Value: {msg.value()}")
                self.message_queue.append(msg.value())
            except KeyboardInterrupt as e:
                print(f"Keyboard Interrupt Received: {e}")
                break
            except Exception as e:
                print(f"Exception {e}")
        consumer.close()
