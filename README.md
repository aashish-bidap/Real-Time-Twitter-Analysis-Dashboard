PREVIEW :





INSTALLATION STEPS :<br>

  **Create network:** <br>
    - $ docker network create kafka-network <br>
  **Spin up the local single-node cluster (will run in the background):**<br>
    - $ docker-compose -f docker-compose.kafka.yml up -d <br>
  **Check the cluster is up and running (wait for "started" to show up):**<br>
    - $ docker-compose -f docker-compose.kafka.yml logs -f broker | grep "started" <br>
  **Start the transaction generator and the fraud detector (will run in the background):**<br>
    - $ docker-compose up -d <br>
  **Create connection between Kafka and ElasticSearch:** <br>
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

**References:**<br>
1.https://github.com/florimondmanca/kafka-fraud-detector <br>
2.https://medium.com/@raymasson/kafka-elasticsearch-connector-fa92a8e3b0bc <br>
3.https://stackoverflow.com/questions/48711455/how-do-i-create-a-dockerized-elasticsearch-index-using-a-python-script-running <br>
4.https://github.com/davidefiocco/dockerized-elasticsearch-indexer/blob/master/indexer/indexer.py <br>
