# Current version (not yet released; still in development)

## Major Features and Improvements

*   Adds support for Python 3.5
*   Initial version of following orchestration platform supported:
    *   Kubeflow
*   Added TensorFlow Model Analysis Colab example
*   Supported split ratio for ExampleGen components.

## Bug fixes and other changes

*   Depends on `apache-beam[gcp]>=2.12,<3`
*   Depends on `tensorflow-data-validation>=0.13.1,<0.14`
*   Depends on `tensorflow-model-analysis>=0.13.2,<0.14`
*   Depends on `tensorflow-transform>=0.13,<0.14`

## Breaking changes
*   Examples code are moved from 'examples' to 'tfx/examples': this ensures that PyPi package contains only one top level python module 'tfx'.

# Version 0.12.0

## Major Features and Improvements

*   Adding TFMA Architecture doc
*   TFX User Guide
*   Initial version of the following TFX components:
    *   CSVExampleGen - CSV data ingestion
    *   BigQueryExampleGen - BigQuery data ingestion
    *   StatisticsGen - calculates statistics for the dataset
    *   SchemaGen - examines the dataset and creates a data schema
    *   ExampleValidator - looks for anomalies and missing values in the dataset
    *   Transform - performs feature engineering on the dataset
    *   Trainer - trains the model
    *   Evaluator - performs analysis of the model performance
    *   ModelValidator - helps validate exported models ensuring that they are
        "good enough" to be pushed to production
    *   Pusher - deploys the model to a serving infrastructure, for example the
        TensorFlow Serving Model Server
*   Initial version of following orchestration platform supported:
    *   Apache Airflow
*   Polished examples based on the Chicago Taxi dataset.

## Bug fixes and other changes

*   Cleanup Colabs to remove TF warnings
*   Performance improvement during shuffling of post-transform data.
*   Changing example to move everything to one file in plugins
*   Adding instructions to refer to README when running Chicago Taxi notebooks

## Breaking changes
