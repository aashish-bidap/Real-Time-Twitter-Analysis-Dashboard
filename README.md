INSTALLATION STEPS <br>
**Create network**
- $ docker network create kafka-network

**Spin up the local single-node Kafka cluster (will run in the background):**
- $ docker-compose -f docker-compose.kafka.yml up -d

**Check the cluster is up and running (wait for "started" to show up):**
- $ docker-compose -f docker-compose.kafka.yml logs -f broker | grep "started"

**Start the transaction generator and the fraud detector (will run in the background):**

- $ docker-compose up -d


**Create connection between Kafka and ElasticSearch:**
curl -X POST -H "Content-Type: application/json" -d '
{
  "name": "test-connector",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "tasks.max": "1",
    "topics": "topic_test",
    "key.ignore": "true",
    "schema.ignore": "true",
    "connection.url": "http://elasticsearch:9200",
    "type.name": "test-type",
    "name": "elasticsearch-sink"
  }
}' localhost:8083/connectors


**References:**

1.https://github.com/florimondmanca/kafka-fraud-detector
2.https://medium.com/@raymasson/kafka-elasticsearch-connector-fa92a8e3b0bc
3.https://stackoverflow.com/questions/48711455/how-do-i-create-a-dockerized-elasticsearch-index-using-a-python-script-running
4.https://github.com/davidefiocco/dockerized-elasticsearch-indexer/blob/master/indexer/indexer.py
