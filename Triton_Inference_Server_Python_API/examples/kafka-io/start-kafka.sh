#!/bin/sh
export DEBIAN_FRONTEND=noninteractive

wget https://dlcdn.apache.org/kafka/3.7.0/kafka_2.13-3.7.0.tgz
tar -xzf kafka_2.13-3.7.0.tgz
cd kafka_2.13-3.7.0

echo "Setting up JAVA 17"
apt-get update -q -y
apt install -q -y openjdk-17-jdk openjdk-17-jre

echo "Configuring brokers to localhost for kafka server"
sed -i -e 's/#listeners=PLAINTEXT:\/\/:9092/listeners=PLAINTEXT:\/\/localhost:9092/g' config/server.properties

echo "Starting zookeeper"
nohup bin/zookeeper-server-start.sh -daemon config/zookeeper.properties > /dev/null 2>&1 &
sleep 5
echo "Successfully started zookeeper, starting kafka brokers"
nohup bin/kafka-server-start.sh -daemon config/server.properties > /dev/null 2>&1 &
sleep 5
echo "Successfully started kafka brokers, creating input and output topics..."

bin/kafka-topics.sh --create --topic inference-input --bootstrap-server localhost:9092
bin/kafka-topics.sh --create --topic inference-output --bootstrap-server localhost:9092

echo "Successfully created topics.\nInput topic: inference-input\nOutput topic: inference-output"

echo "Topic description:"
bin/kafka-topics.sh --describe --topic inference-input --bootstrap-server localhost:9092
bin/kafka-topics.sh --describe --topic inference-output --bootstrap-server localhost:9092
