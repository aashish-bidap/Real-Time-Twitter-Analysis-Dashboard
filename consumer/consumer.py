from kafka import KafkaConsumer
from json import loads
from time import sleep
import json
import os

from kafka import KafkaConsumer, KafkaProducer

KAFKA_BROKER_URL = os.environ.get('KAFKA_BROKER_URL')

consumer = KafkaConsumer(
    'topic_test',
    bootstrap_servers=KAFKA_BROKER_URL,
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='my-group-id',
    value_deserializer=lambda x: json.loads(x),
)
for event in consumer:
    event_data = event.value
    # Do whatever you want
    print(event_data)
    sleep(2)