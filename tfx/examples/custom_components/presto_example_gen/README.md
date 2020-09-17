# Custom TFX Component Example - Presto

# Introduction
This package shows how to compose a custom component in [TensorFlow Extended (TFX)](https://tensorflow.org/tfx). In the following example, a custom ExampleGen component is used to read in data from [Presto](https://prestodb.github.io) using a [Python API](https://github.com/prestodb/presto-python-client/).

## Disclaimer
This package only serves as a demonstration of how to compose a custom component and should not be relied on for production use.

## Prerequisites
* Linux or MacOS
* Python 2.7, 3.5, or 3.6
* Git

### Required packages
* [Apache Beam](https://beam.apache.org/) is used for pipeline orchestration.
* [PrestoPythonClient](https://pypi.org/project/presto-python-client/)
* [TensorFlow](https://tensorflow.org) is used for model training, evaluation and inference.
* [TFX](https://pypi.org/project/tfx/)

# Try It Out

## Step 0: Clone the project and install from source.

```bash
# Supported python version can be found in http://pypi.org/project/tfx.
python -m venv tfx_env
source tfx_env/bin/activate

git clone https://github.com/tensorflow/tfx
git checkout v0.24.0  # Checkout to the latest release.
pip install -e ./tfx  # Install the project in editable mode.
```

ExampleGen's custom configuration protobuf requires a protobuf compiler that is
at least version [3.6.0](http://google.github.io/proto-lens/installing-protoc.html).

## Step 1: Setup Presto on Google Cloud Platform [optional]
Skip this section if a Presto engine is already running. While this example
assumes the following setup, step 3 also demonstrates how to use Presto
ExampleGen for all configurations.

First, install the Presto service on a Cloud Dataproc cluster. Follow this
[tutorial](https://cloud.google.com/dataproc/docs/tutorials/presto-dataproc)
up to and not including the "Prepare data" section.

Instead of using GCP's dataset, [upload](https://cloud.google.com/storage/docs/uploading-objects)
the TFX Chicago Taxi Trips [dataset](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline/data/simple) to the Cloud Storage bucket at `gs://${BUCKET_NAME}/chicago_taxi_trips/csv/`.

Then, create a Hive external table that are backed by the CSV files.

```
gcloud dataproc jobs submit hive \
    --cluster presto-cluster \
    --execute "
        CREATE EXTERNAL TABLE chicago_taxi_trips_csv(
          pickup_community_area INT,
          fare FLOAT,
          trip_start_month INT,
          trip_start_hour INT,
          trip_start_day INT,
          trip_start_timestamp INT,
          pickup_latitude FLOAT,
          pickup_longitude FLOAT,
          dropoff_latitude FLOAT,
          dropoff_longitude FLOAT,
          trip_miles FLOAT,
          pickup_census_tract INT,
          dropoff_census_tract FLOAT,
          payment_type STRING,
          company STRING,
          trip_seconds FLOAT,
          dropoff_community_area FLOAT,
          tips FLOAT)
        ROW FORMAT DELIMITED
        FIELDS TERMINATED BY ','
        STORED AS TEXTFILE
        location 'gs://${BUCKET_NAME}/chicago_taxi_trips/csv/'
        tblproperties ('skip.header.line.count'='1');"
```

Create a Hive external table `chicago_taxi_trips_parquet` that stores the same
data but in Parquet format for better query performance.

```
gcloud dataproc jobs submit hive \
    --cluster presto-cluster \
    --execute "
        CREATE EXTERNAL TABLE chicago_taxi_trips_parquet(
          pickup_community_area INT,
          fare FLOAT,
          trip_start_month INT,
          trip_start_hour INT,
          trip_start_day INT,
          trip_start_timestamp INT,
          pickup_latitude FLOAT,
          pickup_longitude FLOAT,
          dropoff_latitude FLOAT,
          dropoff_longitude FLOAT,
          trip_miles FLOAT,
          pickup_census_tract INT,
          dropoff_census_tract FLOAT,
          payment_type STRING,
          company STRING,
          trip_seconds FLOAT,
          dropoff_community_area FLOAT,
          tips FLOAT)
        STORED AS PARQUET
        location 'gs://${BUCKET_NAME}/chicago_taxi_trips/parquet/';"
```

Load the data from the Hive CSV table into the Hive Parquet table.

```
gcloud dataproc jobs submit hive \
    --cluster presto-cluster \
    --execute "
        INSERT OVERWRITE TABLE chicago_taxi_trips_parquet
        SELECT * FROM chicago_taxi_trips_csv;"
```

Verify that the data loaded correctly. This command should read 15000.

```
gcloud dataproc jobs submit hive \
    --cluster presto-cluster \
    --execute "SELECT COUNT(*) FROM chicago_taxi_trips_parquet;"
```

To verify that the Presto service is available and accessible, follow the steps
outlined under "Run queries" in the GCP [tutorial](https://cloud.google.com/dataproc/docs/tutorials/presto-dataproc#run_queries).

## Step 3: Use Presto ExampleGen
The Presto ExampleGen can be plugged into the pipeline like any other
ExampleGen. The provided example Presto [pipeline](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/example/taxi_pipeline_presto.py) follows closely to the example [pipeline](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_beam.py) that reads from the CSV file. Instead
of CsvExampleGen, the following code snippet is used. The main difference is that a connection configuration is required.

```python
from tfx.examples.custom_components.presto_example_gen.proto import presto_config_pb2
from tfx.examples.custom_components.presto_example_gen.presto_component.component import PrestoExampleGen

presto_config = presto_config_pb2.PrestoConnConfig(host='localhost', port=8080)
example_gen = PrestoExampleGen(presto_config, query='SELECT * FROM chicago_taxi_trips_parquet')
```

The [connection configuration](https://github.com/tensorflow/tfx/blob/master/tfx/examples/custom_components/presto_example_gen/proto/presto_config.proto) is a protobuf that is based off the Presto Python API. Usage may vary depending on how the Presto service is set up.
