from google.cloud import pubsub_v1

from datetime import datetime
import logging
import time
import constants

def main():
    # Simulate 
    logging.getLogger().setLevel(logging.INFO)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(constants.PROJECT_ID, constants.INPUT_TOPIC_NAME)

    my_file = open(constants.TAXI_DATA_FILE, 'r')
    for line in my_file:
        data = line.encode('utf-8')
        future = publisher.publish(
            topic_path, data,
            event_time=datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        )
        print(future.result())
        time.sleep(constants.INPUT_TOPIC_MESSAGE_DELAY)
        
    my_file.close()
    print("Published file.")


if __name__ == "__main__":
    main()
   