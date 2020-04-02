# Command Line Interface for TFX

Note: CLI doesn't have compatibility guarantees, interface might change across
versions

## Introduction

The CLI helps perform full range of pipeline actions like create, update, run,
list and delete pipelines using various orchestrators including Apache Airflow,
Apache Beam and Kubeflow.

## How to use the CLI tool

The CLI is a part of the TFX package. All the commands follow the structure
below:

        tfx <group_name> <sub-command> flags

## Commands

The following command groups are currently supported.

** Important note:

Currently not all commands are supported in every orchestrator. Such commands
explicitly mention the engines supported.

### Pipeline group

The command structure for pipeline group of commands is as follows:

      tfx pipeline <subcommand> <required_flags> [optional_flags]

#### create

Creates a new pipeline in the given orchestrator.

Usage:

      tfx pipeline create <required_flags> [optional_flags]

Required flags:

*   --pipeline_path

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace
*   --package_path
*   --build_target_image
*   --build_base_image
*   --skaffold_cmd

Examples:

Apache Airflow:

    tfx pipeline create \
    --engine=airflow \
    --pipeline_path=<path_to_dsl>

Apache Beam:

    tfx pipeline create \
    --engine=beam \
    --pipeline_path=<path_to_dsl> \

Kubeflow:

    tfx pipeline create \
    --engine=kubeflow \
    --pipeline_path=<path_to_dsl> \
    --package_path=<path_to_package_file> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url> \
    --skaffold_cmd=<path_to_skaffold_binary>

To autodetect engine from user environment, simply avoid using the engine flag
like the example below. For more details, check the flags section.

    tfx pipeline create \
    --pipeline_path=<path_to_dsl>

#### update

Updates an existing pipeline in the given orchestrator.

Usage:

       tfx pipeline update <required_flags> [optional_flags]

Required flags:

*   --pipeline_path

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace
*   --package_path
*   --skaffold_cmd

Examples:

Apache Airflow:

    tfx pipeline update \
    --engine=airflow \
    --pipeline_path=<path_to_dsl>

Apache Beam:

    tfx pipeline update \
    --engine=beam \
    --pipeline_path=<path_to_dsl>

Kubeflow:

    tfx pipeline update \
    --engine=kubeflow \
    --pipeline_path=<path_to_dsl> \
    --package_path=<path_to_package_file> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace>
    --endpoint=<endpoint_url> \
    --skaffold_cmd=<path_to_skaffold_binary>

#### compile

Compiles the pipeline config file to create a workflow file in Kubeflow and
performs the following checks while compiling:

1.  Checks if the pipeline path is valid.
2.  Checks if the pipeline details are extracted successfully from the pipeline
    config file.
3.  Checks if the DagRunner in the pipeline config matches the engine.
4.  Checks if the workflow file is created successfully in the package path
    provided (only for Kubeflow).

Recommended to use before creating or updating a pipeline.

Usage:

        tfx pipeline compile <required_flags> [optional_flags]

Required flags:

*   --pipeline_path

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace
*   --package_path

Examples:

Apache Airflow:

    tfx pipeline compile \
    --engine=airflow \
    --pipeline_path=<path_to_dsl>

Apache Beam:

    tfx pipeline compile \
    --engine=beam \
    --pipeline_path=<path_to_dsl>

Kubeflow:

    tfx pipeline compile \
    --engine=kubeflow \
    --pipeline_path=<path_to_dsl> \
    --package_path=<path_to_package_file> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace>
    --endpoint=<endpoint_url>

#### delete

Deletes a pipeline from the given orchestrator.

Usage:

        tfx pipeline delete <required_flags> [optional_flags]

Required flags:

*   --pipeline_name

Optional flags:

*   --engine
*   --namespace
*   --iap_client_id
*   --endpoint

Examples:

Apache Airflow:

    tfx pipeline delete \
    --engine=airflow \
    --pipeline_name=<name_of_the_pipeline>

Apache Beam:

    tfx pipeline delete \
    --engine=beam \
    --pipeline_name=<name_of_the_pipeline>

Kubeflow:

    tfx pipeline delete \
    --engine=kubeflow \
    --pipeline_name=<name_of the_pipeline> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

#### list

Lists all the pipelines in the given orchestrator.

Usage:

       tfx pipeline list [optional_flags]

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Apache Airflow:

    tfx pipeline list --engine=airflow

Apache Beam:

    tfx pipeline list --engine=beam

Kubeflow:

    tfx pipeline list \
    --engine=kubeflow \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

### Run group

The command structure for run group of commands is as follows:

      tfx run <subcommand> <required_flags> [optional_flags]

#### create

Creates a new run instance for a pipeline in the orchestrator.

Usage:

        tfx run create <required_flags> [optional_flags]

Required flags:

*   --pipeline_name

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Apache Airflow:

    tfx run create \
    --engine=airflow \
    --pipeline_name=<name_of_the_pipeline>

Apache Beam:

    tfx run create \
    --engine=beam \
    --pipeline_name=<name_of_the_pipeline>

Kubeflow:

    tfx run create \
    --engine=kubeflow \
    --pipeline_name=<name_of the_pipeline> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

#### terminate

Stops a run of a given pipeline.

** Important Note: Currently supported only in Kubeflow.

Usage:

       tfx run terminate <required_flags> [optional_flags]

Required flags:

*   --run_id

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Kubeflow:

      tfx run delete \
    --engine=kubeflow \
    --run_id=<run_id> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

#### list

Lists all runs of a pipeline.

** Important Note: Currently not supported in Apache Beam.

Usage:

       tfx run list <required_flags> [optional_flags]

Required flags:

*   --pipeline_name

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Apache Airflow:

    tfx run list \
    --engine=airflow \
    --pipeline_name=<name_of_the_pipeline>

Kubeflow:

    tfx run list \
    --engine=kubeflow \
    --pipeline_name=<name_of the_pipeline> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

#### status

Returns the current status of a run.

** Important Note: Currently not supported in Apache Beam.

Usage:

       tfx run status <required_flags> [optional_flags]

Required flags:

*   --pipeline_name
*   --run_id

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Apache Airflow:

    tfx run status \
    --engine=airflow \
    --run_id=<run_id> \
    --pipeline_name=<name_of_the_pipeline>

Kubeflow:

    tfx run status \
    --engine=kubeflow \
    --run_id=<run_id> \
    --pipeline_name=<name_of the_pipeline> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

#### delete

Deletes a run of a given pipeline.

** Important Note: Currently supported only in Kubeflow

Usage:

       tfx run delete <required_flags> [optional_flags]

Required flags:

*   --run_id

Optional flags:

*   --endpoint
*   --engine
*   --iap_client_id
*   --namespace

Examples:

Kubeflow:

    tfx run delete \
    --engine=kubeflow \
    --run_id=<run_id> \
    --client_id=<IAP_client_id> \
    --namespace=<kubernetes_namespace> \
    --endpoint=<endpoint_url>

### [Experimental] Template group

The command structure for template group of commands is as follows:

      tfx template <subcommand> <required_flags> [optional_flags]

Template is an experimental feature and subject to change at any time.

#### list

List available templates.

Usage:

    tfx template list

#### copy

Copy a template to the destination directory.

Usage:

    tfx template copy <required_flags>

Required flags:

*   --model
*   --pipeline_name
*   --destination_path

Examples:

    tfx template copy \
    --model=<model> \
    --pipeline_name=<name_of_the_pipeline> \
    --destination_path=<path_to_be_created>

## Flags

### Common flags

*   --engine

    The orchestrator to be used for the pipeline. The engine is auto-detected
    based on the environment if not set or else set the flag to one of the
    following values:

    *   airflow: sets engine to Apache Airflow
    *   beam: sets engine to Apache Beam
    *   kubeflow: sets engine to Kubeflow

    ** Important note: The orchestrator required by the DagRunner in the
    pipeline config file must match the selected or autodetected engine. Engine
    auto-detection is based on user environment. If Apache Airflow or Kubeflow
    is not installed then Apache Beam is used by default.

*   --pipeline_name

    The name of the pipeline.

*   --pipeline_path

    The path to the pipeline configuration file.

*   --run_id

    Unique identifier for a run instance of the pipeline.

### Kubeflow specific flags

*   --endpoint

    Endpoint of the KFP API service to connect. If not set, the in-cluster
    service DNS name will be used, which only works if the current environment
    is a pod in the same cluster (such as a Jupyter instance spawned by
    Kubeflow's JupyterHub). If a different connection to cluster exists, such as
    a kubectl proxy connection, then set it to something like
    "127.0.0.1:8080/pipeline". To use an IAP enabled cluster, set it to
    "https://<deployment_name>.endpoints.<project_id>.cloud.goog/pipeline"

*   --iap_client_id

    Client ID for IAP protected endpoint.

*   --namespace

    Kubernetes namespace to connect to the KFP API. Default value is set to
    'kubeflow'.

*   --package_path:

    Path to the pipeline output workflow file. The package_file should end with
    '.tar.gz', '.tgz', '.zip', '.yaml' or '.yml'. When unset, the workflow file
    will be searched in this path:
    "\<current_directory>/\<pipeline_name>\.tar.gz".
