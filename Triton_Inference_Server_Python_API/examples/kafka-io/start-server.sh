#!/bin/sh

export KAFKA_CONSUMER_MAX_WORKER_THREADS=1
export CONSUMER_CONFIGS='{"bootstrap.servers": "localhost:9092", "security.protocol": "PLAINTEXT", "group.id": "triton-server-kafka-consumer"}'
export PRODUCER_CONFIGS='{"bootstrap.servers": "localhost:9092", "security.protocol": "PLAINTEXT"}'
export CONSUMER_TOPICS='inference-input'
export PRODUCER_TOPIC='inference-output'
export MODEL_INPUT_NAME='TEXT'
export MODEL_NAME='tokenizer'
export MODEL_REPOSITORY='./models'

nohup serve run tritonserver_deployment:entrypoint &
tail -f nohup.out
