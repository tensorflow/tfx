# Beam Streaming to ExampleGen E2E Example

## Dependencies
*   TFX current head (needs Date spec PR).
*   GCP instance.
*   Apache Beam with GCP dependencies.

## Overview

This example shows the feasibility of an upstream component from ExampleGen,
which materializes output from a Google Pub/Sub stream into a format which
contains span information and is ingestible by ExampleGen. This specific
example contains 3 parts:

*   `simulate_taxi_data_stream.py` simulates a line by line stream of the 
    Chicago taxi tipping dataset, where each line of the CSV dataset is
    published as a message with `event_time` attribute. The speed of messages
    can be configured under `constants.py`.
*   `materialize_stream.py` is a running beam pipeline which will window
    based on the `event_time` attribute of incoming messages. By default,
    this pipeline has a window of 1 minute and an allowed lateness of one
    minute. Data is materialized in the following folder structure:
    *   `../{YYYY}-{MM}-{DD}/{start_hour}-{start_minute}:{end_hour}-{end_minute}..`
*   `example_gen_pipeline` contains a TFX pipeline with a single ExampleGen
    component to show that the materialized output of the service can ingested.
    This requires the use of date specs, thus the dependency on the Date spec PR.

## Setting up GCP Project
*   Create a Google Cloud Platform project.
*   Enable Cloud Pub/Sub [here](https://cloud.google.com/pubsub/docs/quickstart-console).
*   Create a basic VM instance for setup, under "Compute Engine". Make sure to
    select "Allow full access to all Cloud APIs" under "Identity and API
    access".
*   SSH into the VM instance and run the following:
    *   `sudo apt-get install python3-pip`
    *   `sudo apt-get install git`
    *   `gcloud init` (make sure to set a default location for running from).
    *   `pip install apache-beam[gcp]`
    *   Download this folder onto your VM instance.
*   Create an input topic, either through the command line (by following the
    instructions [here](https://cloud.google.com/pubsub/docs/quickstart-cli))
    or by going to the Pub/Sub menu on the GCP dashboard.
    *   Optionally, create an input topic subscription for debugging.

## How to run this example

*   Open a new `screen` and run `python3 run.py`. This screen will be running
    both the simulated data stream and the materialize beam pipeline.
*   Open a second `screen` and run `python3 example_gen_pipeline.py`. Once a
    a few of the stream files are materialized from the previous streaming service,
    `CsvExampleGen` can ingest these into TFRecords. `example_gen_pipeline.py` 
    contains a pipeline with a single `CsvExampleGen` component.
