
#TODO(developer): replace these.

# Simulating Taxi Data Stream
PROJECT_ID = 'experimental-284817'

INPUT_TOPIC_NAME = 'input_topic'
INPUT_TOPIC_MESSAGE_DELAY = 5

TAXI_DATA_FILE = 'data/data.csv'

# Materializing Taxi Data Stream
OUTPUT_PATH = 'materialized_output_data'

RUNNER_TYPE = 'DirectRunner'

WINDOW_DURATION = 60
WINDOW_ALLOWED_LATENESS = 60

# Example Gen Pipeline
TFX_ROOT = 'tfx/'