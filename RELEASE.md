# Current version (not yet released; still in development)

## Major Features and Improvements

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
    *   ModelValidator - helps validate exported models ensuring that they are "good enough" to be pushed to production
    *   Pusher - deploys the model to a serving infrastructure, for example the TensorFlow Serving Model Server
*   Initial version of following orchestration platform supported:
    *   Apache Airflow
    *   Kubeflow
*   Polished examples based on the Chicago Taxi dataset.

## Bug fixes and other changes
* Performance improvement during shuffling of post-transform data.
* Changing example to move everything to one file in plugins

## Breaking changes

