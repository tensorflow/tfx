# Orchestrating TFX Pipelines

## Apache Airflow

[Apache Airflow](https://airflow.apache.org/) is a platform to
programmatically author, schedule and monitor workflows. TFX uses Airflow to
author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow
scheduler executes tasks on an array of workers while following the specified
dependencies. Rich command line utilities make performing complex surgeries on
DAGs a snap. The rich user interface makes it easy to visualize pipelines
running in production, monitor progress, and troubleshoot issues when needed.
When workflows are defined as code, they become more maintainable, versionable,
testable, and collaborative.

See the [Apache Airflow](https://airflow.apache.org/) for details on installing
and using Apache Airflow.
