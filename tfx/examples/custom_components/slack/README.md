# Custom TFX Component Example - Slack

# Introduction
This example shows how to compose a custom component in [TensorFlow Extended (TFX)](https://tensorflow.org/tfx). We chose to set up a custom
component which uses [Slack](https://slack.com/) [API](https://slack.dev/python-slackclient/)
to integrate a 'human in the loop' step into a TFX pipeline.

This examples uses Slack [RealTime Message](https://api.slack.com/rtm) API which
requires **classic slack bot**.

## Disclaimer
This example only serves as a demonstration of how to compose a custom component
and should not be relied on for productionization use.

## Prerequisites

* Linux or MacOS
* Python 3.6+
* Git

### Required packages
* [Apache Airflow](https://airflow.apache.org/) is used for pipeline orchestration.
* [SlackClient](https://pypi.org/project/slackclient/)
* [TensorFlow](https://tensorflow.org) is used for model training, evaluation and inference.
* [TFX](https://pypi.org/project/tfx/)

### Other prerequisites
* You need to have a Slack account and a channel set up.

# Try It Out

## Common steps

First clone the project and install the source in virtual environment.

```bash
# Supported python version can be found in http://pypi.org/project/tfx.
python -m venv tfx_env
source tfx_env/bin/activate

git clone https://github.com/tensorflow/tfx
git checkout v0.24.0  # Checkout to the latest release.
pip install -e ./tfx
```

This project requires some environment variables that will be used to store
pipeline data and results. Feel free to change the directory below.

```bash
export TAXI_DIR=~/taxi
export TFX_DIR=~/tfx
```

You also need a
[classic slack bot](https://api.slack.com/authentication/migration#classic) with
[bot scope]. After specifying the permission in app configuration page, specify
bot user OAuth access token and channel ID to which the bot will ask for user
approval.

```bash
export TFX_SLACK_BOT_TOKEN="xoxb-..."
export TFX_SLACK_CHANNEL_ID="C..."
```

## Running Pipeline Locally

In order to run pipeline locally, you need to additionally install airflow in
the virtual environment and configure some variables.

```bash
# Inside virtual environment
pip install apache-airflow
export AIRFLOW_HOME=~/airflow  # Feel free to choose other directory.
airflow initdb
```

The benefit of the local example is that you can edit any part of the pipeline
and experiment very quickly with various components. The example comes with a
small subset of the Taxi Trips dataset as CSV files.

Let's copy the dataset CSV to the directory where TFX ExampleGen will ingest it
from:

```bash
mkdir -p $TAXI_DIR/data/simple
cp data/simple/data.csv $TAXI_DIR/data/simple
```

Let's copy the TFX pipeline definition to Airflow's
`DAGs directory` `($AIRFLOW_HOME/dags)` so it can run the pipeline:

```bash
mkdir $AIRFLOW_HOME/dags/
cp example/taxi_pipeline_slack.py $AIRFLOW_HOME/dags/
```

The module file `taxi_utils_slack.py` used by the Trainer and Transform
components will reside in `$TAXI_DIR`, let's copy it there:

```bash
cp example/taxi_utils_slack.py $TAXI_DIR
```

Follow similar steps in
[*Run the local example*](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline#run-the-local-example)
in our regular chicago_taxi_pipeline example to run the pipeline. Just note that
the pipeline name is `chicago_taxi_slack`.

## Running Pipeline in Google Cloud Platform

Once you've done playing with your pipeline locally, you can easily deploy the
pipeline in Google Cloud Platform for production.

First prepare a gcs bucket for the pipeline run root:

```bash
gsutil mb -p ${PROJECT_ID} gs://${BUCKET_NAME}
```

Let's copy the dataset CSV to the GCS where TFX ExampleGen will ingest it
from:

```bash
cp data/simple/data.csv gs://${BUCKET_NAME}/data/simple/
```

Let's copy the TFX pipeline definition to the root of the slack example
and update the _input_bucket/_update_bucket to gs://${BUCKET_NAME}:

```bash
cp example/taxi_pipeline_slack_kubeflow.py ./
```

Compile the slack example. Under the hood, tfx CLI creates a container with the
slack component installed using Skaffold and calls kubeflow dag runner:

```bash
tfx pipeline create --engine kubeflow --build_target_image ${target_image_name} \
  --pipeline_path taxi_pipeline_slack_kubeflow.py
```

Finally upload the generated `chicago_taxi_slack.tar.gz` to the Kubeflow
Pipeline UI. Remember to input the pipeline-root to `gs://${BUCKET_NAME}`.


## Interact with Slack

After the `Model Validator` phase succeeds, you will get a Slack message sent to
your Slack channel asking you to review a model with a URI. If you reply 'LGTM'
or 'approve' (**in thread**), the pipeline will continue to push the model. If
you reply 'decline' or 'reject' (**in thread**), the pipeline will continue
without pushing the model. All the key words are case insensitive. If your reply
fall out of the above keywords, you will get a hint.
