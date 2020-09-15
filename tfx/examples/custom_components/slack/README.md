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
While it is not mandantory, we highly recommend trying out the example in a virtual environment. You can achieve that by the following:

```bash
cd
python -m virtualenv -p python3.6 tfx_env
source tfx_env/bin/activate
```

## Step 0: Setup Environment
First install required packages:

```bash
pip install tensorflow==1.13.1
pip install apache-airflow
pip install docker
```

Configure common paths:

```bash
export AIRFLOW_HOME=~/airflow
export TAXI_DIR=~/taxi
export TFX_DIR=~/tfx
```

Initialize Airflow:

```bash
airflow initdb
```

## Step 1: Setup a Slack app
Follow this Slack [tutorial](https://github.com/slackapi/python-slackclient/blob/master/tutorial/01-creating-the-slack-app.md) or similar one to set up a slack app. After setup, set the `Bot User OAuth Access Token` in your environment.

```bash
export SLACK_BOT_TOKEN={your_token}
```

## Step 2: Install Custom SlackComponent

```bash
cd
git clone https://github.com/tensorflow/tfx.git

cd tfx/tfx/examples/custom_components/slack
pip install -e .
```

## Step 3: Try Out Example
### Copy the pipeline definition to Airflow's DAG directory (Local)

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

You will also need to change the `slack_channel_id` field to your own Slack channel id
in the pipeline definition:

```bash
sed -i 's/my-channel-id/{your-channel-id}/g' $AIRFLOW_HOME/dags/taxi_pipeline_slack.py
```

The module file `taxi_utils_slack.py` used by the Trainer and Transform
components will reside in `$TAXI_DIR`, let's copy it there:

```bash
cp example/taxi_utils_slack.py $TAXI_DIR
```

### Run the pipeline (Local)
Follow similar steps in [Run the local example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline#run-the-local-example)
in our regular chicago_taxi_pipeline example to run the pipeline. Just note that
the pipeline name is `chicago_taxi_slack`.

### Compile the pipeline (GCP)
Prepare a gcs bucket for the pipeline run root:

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

You will also need to change the `slack_channel_id` field to your own Slack
channel id in the pipeline definition:

```bash
sed -i 's/my-channel-id/{your-channel-id}/g' ./taxi_pipeline_slack_kubeflow.py
```

Compile the slack example. Under the hood, tfx CLI creates a container with the
slack component installed using Skaffold and calls kubeflow dag runner:
```bash
tfx pipeline create --engine kubeflow --build_target_image ${target_image_name} \
  --pipeline_path taxi_pipeline_slack_kubeflow.py
```

### Run the pipeline (GCP)
Upload the generated chicago_taxi_slack.tar.gz and experiment in the Kubeflow
Pipeline UI. Remember to input the pipeline-root to gs://${BUCKET_NAME}.


### Interact with Slack
After the `Model Validator` phase succeeds, you will get a Slack message sent to
your Slack channel asking you to review a model with a URI. If you reply 'LGTM'
or 'approve' (**in thread**), the pipeline will continue to push the model. If
you reply 'decline' or 'reject' (**in thread**), the pipeline will continue
without pushing the model. All the key words are case insensitive. If your reply
fall out of the above keywords, you will get a hint.
