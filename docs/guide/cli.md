# Using the TFX Command-line Interface

The TFX command-line interface (CLI) performs a full range of pipeline actions
using pipeline orchestrators, such as Kubeflow Pipelines, Vertex Pipelines.
Local orchestrator can be also used for faster development or debugging. Apache
Beam and Apache airflow is supported as experimental features. For example, you
can use the CLI to:

*   Create, update, and delete pipelines.
*   Run a pipeline and monitor the run on various orchestrators.
*   List pipelines and pipeline runs.

!!! Note
    The TFX CLI doesn't currently provide compatibility guarantees. The CLI
    interface might change as new versions are released.

## About the TFX CLI

The TFX CLI is installed as a part of the TFX package. All CLI commands follow
the structure below:

```bash
tfx <command-group> <command> <flags>
```

The following command-group options are currently supported:

*   [`tfx pipeline`](#tfx-pipeline) - Create and manage TFX pipelines.
*   [`tfx run`](#tfx-run) - Create and manage runs of TFX pipelines on various
    orchestration platforms.
*   [`tfx template`](#tfx-template-experimental) - Experimental commands for
    listing and copying TFX pipeline templates.

Each command group provides a set of commands. Follow the
instructions in the [pipeline commands](#tfx-pipeline),
[run commands](#tfx-run), and [template commands](#tfx-template-experimental)
sections to learn more about using these commands.

!!! Warning
    Currently not all commands are supported in every orchestrator. Such
    commands explicitly mention the engines supported.

Flags let you pass arguments into CLI commands. Words in flags are separated
with either a hyphen (`-`) or an underscore (`_`). For example, the pipeline
name flag can be specified as either `--pipeline-name` or `--pipeline_name`.
This document specifies flags with underscores for brevity. Learn more about
[flags used in the TFX CLI](#understanding-tfx-cli-flags).

## tfx pipeline

The structure for commands in the `tfx pipeline` command group is as follows:

```bash
tfx pipeline command required-flags [optional-flags]
```

Use the following sections to learn more about the commands in the `tfx
pipeline` command group.

### create

Creates a new pipeline in the given orchestrator.

Usage:

```bash
tfx pipeline create --pipeline_path=pipeline-path [--endpoint=endpoint --engine=engine \
--iap_client_id=iap-client-id --namespace=namespace \
--build_image --build_base_image=build-base-image]
```

\--pipeline\_path=`pipeline-path`{.variable}
:   The path to the pipeline configuration file.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint when using Kubeflow Pipelines.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.

\--build\_image

:   (Optional.) When the `engine`{.variable} is **kubeflow** or **vertex**, TFX creates a container image for your pipeline if specified. `Dockerfile` in the current directory will be used, and TFX will automatically generate one if not exists.

    The built image will be pushed to the remote registry which is specified in `KubeflowDagRunnerConfig` or `KubeflowV2DagRunnerConfig`.

\--build\_base\_image=`build-base-image`{.variable}

:   (Optional.) When the `engine`{.variable} is **kubeflow**, TFX creates a container image for your pipeline. The build base image specifies the base container image to use when building the pipeline container image.


#### Examples

Kubeflow:

```bash
tfx pipeline create --engine=kubeflow --pipeline_path=pipeline-path \
--iap_client_id=iap-client-id --namespace=namespace --endpoint=endpoint \
--build_image
```

Local:

```bash
tfx pipeline create --engine=local --pipeline_path=pipeline-path
```

Vertex:

```bash
tfx pipeline create --engine=vertex --pipeline_path=pipeline-path \
--build_image
```

To autodetect engine from user environment, simply avoid using the engine flag
like the example below. For more details, check the flags section.

```bash
tfx pipeline create --pipeline_path=pipeline-path
```

### update

Updates an existing pipeline in the given orchestrator.

Usage:

```bash
tfx pipeline update --pipeline_path=pipeline-path [--endpoint=endpoint --engine=engine \
--iap_client_id=iap-client-id --namespace=namespace --build_image]
```

\--pipeline\_path=`pipeline-path`{.variable}
:   The path to the pipeline configuration file.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.

\--build\_image

:   (Optional.) When the `engine`{.variable} is **kubeflow** or **vertex**, TFX creates a container image for your pipeline if specified. `Dockerfile` in the current directory will be used.

    The built image will be pushed to the remote registry which is specified in `KubeflowDagRunnerConfig` or `KubeflowV2DagRunnerConfig`.


#### Examples

Kubeflow:

```bash
tfx pipeline update --engine=kubeflow --pipeline_path=pipeline-path \
--iap_client_id=iap-client-id --namespace=namespace --endpoint=endpoint \
--build_image
```

Local:

```bash
tfx pipeline update --engine=local --pipeline_path=pipeline-path
```

Vertex:

```bash
tfx pipeline update --engine=vertex --pipeline_path=pipeline-path \
--build_image
```

### compile

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

```bash
tfx pipeline compile --pipeline_path=pipeline-path [--engine=engine]
```

\--pipeline\_path=`pipeline-path`{.variable}
:   The path to the pipeline configuration file.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.


#### Examples

Kubeflow:

```bash
tfx pipeline compile --engine=kubeflow --pipeline_path=pipeline-path
```

Local:

```bash
tfx pipeline compile --engine=local --pipeline_path=pipeline-path
```

Vertex:

```bash
tfx pipeline compile --engine=vertex --pipeline_path=pipeline-path
```

### delete

Deletes a pipeline from the given orchestrator.

Usage:

```bash
tfx pipeline delete --pipeline_path=pipeline-path [--endpoint=endpoint --engine=engine \
--iap_client_id=iap-client-id --namespace=namespace]
```

\--pipeline\_path=`pipeline-path`{.variable}
:   The path to the pipeline configuration file.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


#### Examples

Kubeflow:

```bash
tfx pipeline delete --engine=kubeflow --pipeline_name=pipeline-name \
--iap_client_id=iap-client-id --namespace=namespace --endpoint=endpoint
```

Local:

```bash
tfx pipeline delete --engine=local --pipeline_name=pipeline-name
```

Vertex:

```bash
tfx pipeline delete --engine=vertex --pipeline_name=pipeline-name
```

### list

Lists all the pipelines in the given orchestrator.

Usage:

```bash
tfx pipeline list [--endpoint=endpoint --engine=engine \
--iap_client_id=iap-client-id --namespace=namespace]
```

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


#### Examples

Kubeflow:

```bash
tfx pipeline list --engine=kubeflow --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint
```

Local:

```bash
tfx pipeline list --engine=local
```

Vertex:

```bash
tfx pipeline list --engine=vertex
```

## tfx run

The structure for commands in the `tfx run` command group is as follows:

```bash
tfx run command required-flags [optional-flags]
```

Use the following sections to learn more about the commands in the `tfx run`
command group.

### create

Creates a new run instance for a pipeline in the orchestrator. For Kubeflow, the
most recent pipeline version of the pipeline in the cluster is used.

Usage:

```bash
tfx run create --pipeline_name=pipeline-name [--endpoint=endpoint \
--engine=engine --iap_client_id=iap-client-id --namespace=namespace]
```

\--pipeline\_name=`pipeline-name`{.variable}
:   The name of the pipeline.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--runtime\_parameter=`parameter-name`{.variable}=`parameter-value`{.variable}
:   (Optional.) Sets a runtime parameter value. Can be set multiple times to set values of multiple variables. Only applicable to `airflow`, `kubeflow` and `vertex` engine.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.

\--project=`GCP-project-id`{.variable}
:   (Required for Vertex.) GCP project id for the vertex pipeline.

\--region=`GCP-region`{.variable}
:   (Required for Vertex.) GCP region name like us-central1. See \[Vertex documentation\](https://cloud.google.com/vertex-ai/docs/general/locations) for available regions.


#### Examples

Kubeflow:

```bash
tfx run create --engine=kubeflow --pipeline_name=pipeline-name --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint
```

Local:

```bash
tfx run create --engine=local --pipeline_name=pipeline-name
```

Vertex:

```bash
tfx run create --engine=vertex --pipeline_name=pipeline-name \
  --runtime_parameter=var_name=var_value \
  --project=gcp-project-id --region=gcp-region
```

### terminate

Stops a run of a given pipeline.

!!! note "Important Note"
    Currently supported only in Kubeflow.

Usage:

```bash
tfx run terminate --run_id=run-id [--endpoint=endpoint --engine=engine \
--iap_client_id=iap-client-id --namespace=namespace]
```

\--run\_id=`run-id`{.variable}
:   Unique identifier for a pipeline run.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


#### Examples

Kubeflow:

```bash
tfx run delete --engine=kubeflow --run_id=run-id --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint
```

### list

Lists all runs of a pipeline.

!!! note "Important Note"
    Currently not supported in Local and Apache Beam.

Usage:

```bash
tfx run list --pipeline_name=pipeline-name [--endpoint=endpoint \
--engine=engine --iap_client_id=iap-client-id --namespace=namespace]
```

\--pipeline\_name=`pipeline-name`{.variable}
:   The name of the pipeline.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **airflow**: (experimental) sets engine to Apache Airflow

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.

#### Examples

Kubeflow:

```bash
tfx run list --engine=kubeflow --pipeline_name=pipeline-name --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint
```

### status

Returns the current status of a run.

!!! note "Important Note"
    Currently not supported in Local and Apache Beam.

Usage:

```bash
tfx run status --pipeline_name=pipeline-name --run_id=run-id [--endpoint=endpoint \
--engine=engine --iap_client_id=iap-client-id --namespace=namespace]
```

\--pipeline\_name=`pipeline-name`{.variable}
:   The name of the pipeline.

\--run\_id=`run-id`{.variable}
:   Unique identifier for a pipeline run.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **airflow**: (experimental) sets engine to Apache Airflow

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


#### Examples

Kubeflow:

```bash
tfx run status --engine=kubeflow --run_id=run-id --pipeline_name=pipeline-name \
--iap_client_id=iap-client-id --namespace=namespace --endpoint=endpoint
```

### delete

Deletes a run of a given pipeline.

!!! note Important Note
    Currently supported only in Kubeflow

Usage:

```bash
tfx run delete --run_id=run-id [--engine=engine --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint]
```

\--run\_id=`run-id`{.variable}
:   Unique identifier for a pipeline run.

\--endpoint=`endpoint`{.variable}

:   (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--engine=`engine`{.variable}

:   (Optional.) The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
      The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--iap\_client\_id=`iap-client-id`{.variable}
:   (Optional.) Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


#### Examples

Kubeflow:

```bash
tfx run delete --engine=kubeflow --run_id=run-id --iap_client_id=iap-client-id \
--namespace=namespace --endpoint=endpoint
```

## tfx template [Experimental]

The structure for commands in the `tfx template` command group is as follows:

```bash
tfx template command required-flags [optional-flags]
```

Use the following sections to learn more about the commands in the `tfx
template` command group. Template is an experimental feature and subject to
change at any time.

### list

List available TFX pipeline templates.

Usage:

```bash
tfx template list
```

### copy

Copy a template to the destination directory.

Usage:

```bash
tfx template copy --model=model --pipeline_name=pipeline-name \
--destination_path=destination-path
```

\--model=`model`{.variable}
:   The name of the model built by the pipeline template.

\--pipeline\_name=`pipeline-name`{.variable}
:   The name of the pipeline.

\--destination\_path=`destination-path`{.variable}
:   The path to copy the template to.


## Understanding TFX CLI Flags

### Common flags

\--engine=`engine`{.variable}

:   The orchestrator to be used for the pipeline. The value of engine must match on of the following values:

    -   **kubeflow**: sets engine to Kubeflow
    -   **local**: sets engine to local orchestrator
    -   **vertex**: sets engine to Vertex Pipelines
    -   **airflow**: (experimental) sets engine to Apache Airflow
    -   **beam**: (experimental) sets engine to Apache Beam

    If the engine is not set, the engine is auto-detected based on the environment.

    !!! note "Important Note"
        The orchestrator required by the DagRunner in the pipeline config file must match the selected or autodetected engine. Engine auto-detection is based on user environment. If Apache Airflow and Kubeflow Pipelines are not installed, then the local orchestrator is used by default.

\--pipeline\_name=`pipeline-name`{.variable}
:   The name of the pipeline.

\--pipeline\_path=`pipeline-path`{.variable}
:   The path to the pipeline configuration file.

\--run\_id=`run-id`{.variable}
:   Unique identifier for a pipeline run.


### Kubeflow specific flags

\--endpoint=`endpoint`{.variable}

:   Endpoint of the Kubeflow Pipelines API service. The endpoint of your Kubeflow Pipelines API service is the same as URL of the Kubeflow Pipelines dashboard. Your endpoint value should be something like:

        https://host-name/pipeline

    If you do not know the endpoint for your Kubeflow Pipelines cluster, contact you cluster administrator.

    If the `--endpoint` is not specified, the in-cluster service DNS name is used as the default value. This name works only if the CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a [Kubeflow Jupyter notebooks](https://www.kubeflow.org/docs/components/notebooks/jupyter-tensorflow-examples/){.external} instance.

\--iap\_client\_id=`iap-client-id`{.variable}
:   Client ID for IAP protected endpoint.

\--namespace=`namespace`{.variable}
:   Kubernetes namespace to connect to the Kubeflow Pipelines API. If the namespace is not specified, the value defaults to `kubeflow`.


## Generated files by TFX CLI

When pipelines are created and run, several files are generated for pipeline
management.

-   ${HOME}/tfx/local, beam, airflow, vertex
    -   Pipeline metadata read from the configuration is stored under
        `${HOME}/tfx/${ORCHESTRATION_ENGINE}/${PIPELINE_NAME}`. This location
        can be customized by setting environment varaible like `AIRFLOW_HOME` or
        `KUBEFLOW_HOME`. This behavior might be changed in future releases. This
        directory is used to store pipeline information including pipeline ids
        in the Kubeflow Pipelines cluster which is needed to create runs or
        update pipelines.
    -   Before TFX 0.25, these files were located under
        `${HOME}/${ORCHESTRATION_ENGINE}`. In TFX 0.25, files in the old
        location will be moved to the new location automatically for smooth
        migration.
    -   From TFX 0.27, kubeflow doesn't create these metadata files in local
        filesystem. However, see below for other files that kubeflow creates.
-   (Kubeflow only) Dockerfile and a container image
    -   Kubeflow Pipelines requires two kinds of input for a pipeline. These
        files are generated by TFX in the current directory.
    -   One is a container image which will be used to run components in the
        pipeline. This container image is built when a pipeline for Kubeflow
        Pipelines is created or updated with `--build-image` flag. TFX CLI will
        generate `Dockerfile` if not exists, and will build and push a container
        image to the registry specified in KubeflowDagRunnerConfig.
