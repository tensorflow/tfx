# Using the TFX Command-line Interface

The TFX command-line interface (CLI) performs a full range of pipeline actions
using pipeline orchestrators, such as Apache Airflow, Apache Beam, and Kubeflow
Pipelines. For example, you can use the CLI to:

*   Create, update, and delete pipelines.
*   Run a pipeline and monitor the run on various orchestrators.
*   List pipelines and pipeline runs.

Note: The TFX CLI doesn't currently provide compatibility guarantees. The CLI
interface might change as new versions are released.

## About the TFX CLI

The TFX CLI is installed as a part of the TFX package. All CLI commands follow
the structure below:

<pre class="devsite-terminal">
tfx <var>command-group</var> <var>command</var> <var>flags</var>
</pre>

The following <var>command-group</var> options are currently supported:

*   [tfx pipeline](#tfx-pipeline) - Create and manage TFX pipelines.
*   [tfx run](#tfx-run) - Create and manage runs of TFX pipelines on various
    orchestration platforms.
*   [tfx template](#tfx-template-experimental) - Experimental commands for
    listing and copying TFX pipeline templates.

Each command group provides a set of <var>commands</var>. Follow the
instructions in the [pipeline commands](#tfx-pipeline),
[run commands](#tfx-run), and [template commands](#tfx-template-experimental)
sections to learn more about using these commands.

Warning: Currently not all commands are supported in every orchestrator. Such
commands explicitly mention the engines supported.

Flags let you pass arguments into CLI commands. Words in flags are separated
with either a hyphen (`-`) or an underscore (`_`). For example, the pipeline
name flag can be specified as either `--pipeline-name` or `--pipeline_name`.
This document specifies flags with underscores for brevity. Learn more about
[<var>flags</var> used in the TFX CLI](#understanding-tfx-cli-flags).

## tfx pipeline

The structure for commands in the `tfx pipeline` command group is as follows:

<pre class="devsite-terminal">
tfx pipeline <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

Use the following sections to learn more about the commands in the `tfx
pipeline` command group.

### create

Creates a new pipeline in the given orchestrator.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline create --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var> \
--build_target_image=<var>build-target-image</var> --build_base_image=<var>build-base-image</var> \
--skaffold_cmd=<var>skaffold-command</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var></dt>
  <dd>The path to the pipeline configuration file.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>

  <dt>--package_path=<var>package-path</var></dt>
  <dd>
    <p>
      (Optional.) Path to the compiled pipeline as a file. The compiled pipeline
      should be a compressed file (<code>.tar.gz</code>, <code>.tgz</code>, or
      <code>.zip</code>) or a YAML file (<code>.yaml</code> or
      <code>.yml</code>).
    </p>
    <p>
      If <var>package-path</var> is not specified, TFX uses the following as
      the default path:
      <code><var>current_directory</var>/<var>pipeline_name</var>.tar.gz</code>
    </p>
  </dd>
  <dt>--build_target_image=<var>build-target-image</var></dt>
  <dd>
    <p>
      (Optional.) When the <var>engine</var> is <strong>kubeflow</strong>, TFX
      creates a container image for your pipeline. The build target image
      specifies the name, container image repository, and tag to use when
      creating the pipeline container image. If you do not specify a tag, the
      container image is tagged as <code>latest</code>.
    </p>
    <p>
      For your Kubeflow Pipelines cluster to run your pipeline, the cluster must
      be able to access the specified container image repository.
    </p>
  </dd>
  <dt>--build_base_image=<var>build-base-image</var></dt>
  <dd>
    <p>
      (Optional.) When the <var>engine</var> is <strong>kubeflow</strong>, TFX
      creates a container image for your pipeline. The build base image
      specifies the base container image to use when building the pipeline
      container image.
    </p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var></dt>
  <dd>
    <p>
      (Optional.) The path to <a href="https://skaffold.dev/" class="external">
      Skaffold</a> on your computer.
    </p>
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx pipeline create --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx pipeline create --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline create --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var> \
--skaffold_cmd=<var>skaffold-cmd</var>
</pre>

To autodetect engine from user environment, simply avoid using the engine flag
like the example below. For more details, check the flags section.

<pre class="devsite-terminal">
tfx pipeline create --pipeline_path=<var>pipeline-path</var> --endpoint --iap_client_id --namespace \
--package_path --skaffold_cmd
</pre>

### update

Updates an existing pipeline in the given orchestrator.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline update --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var> \
--skaffold_cmd=<var>skaffold-command</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var></dt>
  <dd>The path to the pipeline configuration file.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>

  <dt>--package_path=<var>package-path</var></dt>
  <dd>
    <p>
      (Optional.) Path to the compiled pipeline as a file. The compiled pipeline
      should be a compressed file (<code>.tar.gz</code>, <code>.tgz</code>, or
      <code>.zip</code>) or a YAML file (<code>.yaml</code> or
      <code>.yml</code>).
    </p>
    <p>
      If <var>package-path</var> is not specified, TFX uses the following as
      the default path:
      <code><var>current_directory</var>/<var>pipeline_name</var>.tar.gz</code>
    </p>
  </dd>
  <dt>--skaffold_cmd=<var>skaffold-cmd</var></dt>
  <dd>
    <p>
      (Optional.) The path to <a href="https://skaffold.dev/" class="external">
      Skaffold</a> on your computer.
    </p>
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx pipeline update --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx pipeline update --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline update --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var> \
--skaffold_cmd=<var>skaffold-cmd</var>
</pre>

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

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline compile --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --package_path=<var>package-path</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var></dt>
  <dd>The path to the pipeline configuration file.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>

  <dt>--package_path=<var>package-path</var></dt>
  <dd>
    <p>
      (Optional.) Path to the compiled pipeline as a file. The compiled pipeline
      should be a compressed file (<code>.tar.gz</code>, <code>.tgz</code>, or
      <code>.zip</code>) or a YAML file (<code>.yaml</code> or
      <code>.yml</code>).
    </p>
    <p>
      If <var>package-path</var> is not specified, TFX uses the following as
      the default path:
      <code><var>current_directory</var>/<var>pipeline_name</var>.tar.gz</code>
    </p>
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx pipeline compile --engine=airflow --pipeline_path=<var>pipeline-path</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx pipeline compile --engine=beam --pipeline_path=<var>pipeline-path</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline compile --engine=kubeflow --pipeline_path=<var>pipeline-path</var> --package_path=<var>package-path</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### delete

Deletes a pipeline from the given orchestrator.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline delete --pipeline_path=<var>pipeline-path</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_path=<var>pipeline-path</var></dt>
  <dd>The path to the pipeline configuration file.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx pipeline delete --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx pipeline delete --engine=beam --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline delete --engine=kubeflow --pipeline_name=<var>pipeline-name</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### list

Lists all the pipelines in the given orchestrator.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx pipeline list [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx pipeline list --engine=airflow
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx pipeline list --engine=beam
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx pipeline list --engine=kubeflow --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

## tfx run

The structure for commands in the `tfx run` command group is as follows:

<pre class="devsite-terminal">
tfx run <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

Use the following sections to learn more about the commands in the `tfx run`
command group.

### create

Creates a new run instance for a pipeline in the orchestrator.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run create --pipeline_name=<var>pipeline-name</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var></dt>
  <dd>The name of the pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx run create --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Apache Beam:

<pre class="devsite-terminal">
tfx run create --engine=beam --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx run create --engine=kubeflow --pipeline_name=<var>pipeline-name</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### terminate

Stops a run of a given pipeline.

** Important Note: Currently supported only in Kubeflow.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run terminate --run_id=<var>run-id</var> [--endpoint=<var>endpoint</var> --engine=<var>engine</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var></dt>
  <dd>Unique identifier for a pipeline run.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=<var>run-id</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### list

Lists all runs of a pipeline.

** Important Note: Currently not supported in Apache Beam.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run list --pipeline_name=<var>pipeline-name</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var></dt>
  <dd>The name of the pipeline.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx run list --engine=airflow --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx run list --engine=kubeflow --pipeline_name=<var>pipeline-name</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### status

Returns the current status of a run.

** Important Note: Currently not supported in Apache Beam.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run status --pipeline_name=<var>pipeline-name</var> --run_id=<var>run-id</var> [--endpoint=<var>endpoint</var> \
--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var>]
</pre>

<dl>
  <dt>--pipeline_name=<var>pipeline-name</var></dt>
  <dd>The name of the pipeline.</dd>
  <dt>--run_id=<var>run-id</var></dt>
  <dd>Unique identifier for a pipeline run.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Apache Airflow:

<pre class="devsite-terminal">
tfx run status --engine=airflow --run_id=<var>run-id</var> --pipeline_name=<var>pipeline-name</var>
</pre>

Kubeflow:

<pre class="devsite-terminal">
tfx run status --engine=kubeflow --run_id=<var>run-id</var> --pipeline_name=<var>pipeline-name</var> \
--iap_client_id=<var>iap-client-id</var> --namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

### delete

Deletes a run of a given pipeline.

** Important Note: Currently supported only in Kubeflow

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx run delete --run_id=<var>run-id</var> [--engine=<var>engine</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>]
</pre>

<dl>
  <dt>--run_id=<var>run-id</var></dt>
  <dd>Unique identifier for a pipeline run.</dd>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      (Optional.) Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      (Optional.) The orchestrator to be used for the pipeline. The value of
      engine must match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>
  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    (Optional.) Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    (Optional.) Kubernetes namespace to connect to the Kubeflow Pipelines API.
    If the namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>
</dl>

#### Examples:

Kubeflow:

<pre class="devsite-terminal">
tfx run delete --engine=kubeflow --run_id=<var>run-id</var> --iap_client_id=<var>iap-client-id</var> \
--namespace=<var>namespace</var> --endpoint=<var>endpoint</var>
</pre>

## tfx template [Experimental]

The structure for commands in the `tfx template` command group is as follows:

<pre class="devsite-terminal">
tfx template <var>command</var> <var>required-flags</var> [<var>optional-flags</var>]
</pre>

Use the following sections to learn more about the commands in the `tfx
template` command group. Template is an experimental feature and subject to
change at any time.

### list

List available TFX pipeline templates.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template list
</pre>

### copy

Copy a template to the destination directory.

Usage:

<pre class="devsite-click-to-copy devsite-terminal">
tfx template copy --model=<var>model</var> --pipeline_name=<var>pipeline-name</var> \
--destination_path=<var>destination-path</var>
</pre>

<dl>
  <dt>--model=<var>model</var></dt>
  <dd>The name of the model built by the pipeline template.</dd>
  <dt>--pipeline_name=<var>pipeline-name</var></dt>
  <dd>The name of the pipeline.</dd>
  <dt>--destination_path=<var>destination-path</var></dt>
  <dd>The path to copy the template to.</dd>
</dl>

## Understanding TFX CLI Flags

### Common flags

<dl>
  <dt>--engine=<var>engine</var></dt>
  <dd>
    <p>
      The orchestrator to be used for the pipeline. The value of engine must
      match on of the following values:
    </p>
    <ul>
      <li><strong>airflow</strong>: sets engine to Apache Airflow</li>
      <li><strong>beam</strong>: sets engine to Apache Beam</li>
      <li><strong>kubeflow</strong>: sets engine to Kubeflow</li>
    </ul>
    <p>
      If the engine is not set, the engine is auto-detected based on the
      environment.
    </p>
    <p>
      ** Important note: The orchestrator required by the DagRunner in the
      pipeline config file must match the selected or autodetected engine.
      Engine auto-detection is based on user environment. If Apache Airflow
      and Kubeflow Pipelines are not installed, then Apache Beam is used by
      default.
    </p>
  </dd>

  <dt>--pipeline_name=<var>pipeline-name</var></dt>
  <dd>The name of the pipeline.</dd>

  <dt>--pipeline_path=<var>pipeline-path</var></dt>
  <dd>The path to the pipeline configuration file.</dd>

  <dt>--run_id=<var>run-id</var></dt>
  <dd>Unique identifier for a pipeline run.</dd>

</dl>

### Kubeflow specific flags

<dl>
  <dt>--endpoint=<var>endpoint</var></dt>
  <dd>
    <p>
      Endpoint of the Kubeflow Pipelines API service. The endpoint
      of your Kubeflow Pipelines API service is the same as URL of the Kubeflow
      Pipelines dashboard. Your endpoint value should be something like:
    </p>

    <pre>https://<var>host-name</var>/pipeline</pre>

    <p>
      If you do not know the endpoint for your Kubeflow Pipelines cluster,
      contact you cluster administrator.
    </p>

    <p>
      If the <code>--endpoint</code> is not specified, the in-cluster service
      DNS name is used as the default value. This name works only if the
      CLI command executes in a pod on the Kubeflow Pipelines cluster, such as a
      <a href="https://www.kubeflow.org/docs/notebooks/why-use-jupyter-notebook/"
           class="external">Kubeflow Jupyter notebooks</a> instance.
    </p>
  </dd>

  <dt>--iap_client_id=<var>iap-client-id</var></dt>
  <dd>
    Client ID for IAP protected endpoint.
  </dd>

  <dt>--namespace=<var>namespace</var>
  <dd>
    Kubernetes namespace to connect to the Kubeflow Pipelines API. If the
    namespace is not specified, the value defaults to
    <code>kubeflow</code>.
  </dd>

  <dt>--package_path=<var>package-path</var></dt>
  <dd>
    <p>
      Path to the compiled pipeline as a file. The compiled pipeline should be a
      compressed file (<code>.tar.gz</code>, <code>.tgz</code>, or
      <code>.zip</code>) or a YAML file (<code>.yaml</code> or
      <code>.yml</code>).
    </p>
    <p>
      If <var>package-path</var> is not specified, TFX uses the following as
      the default path:
      <code><var>current_directory</var>/<var>pipeline_name</var>.tar.gz</code>
    </p>
  </dd>

</dl>
