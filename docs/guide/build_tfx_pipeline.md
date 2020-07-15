# Building a TFX Pipeline

TFX makes it easier to orchestrate your machine learning (ML) workflow as
pipeline, in order to:

*   Automate your ML process, which lets you regularly retrain, evaluate, and
    deploy your model.
*   Create ML pipelines which include deep analysis of model performance and
    validation of newly trained models to ensure performance and reliability.
*   Monitor training data for anomalies and eliminate training-serving skew
*   Increase the velocity of experimentation by running a pipeline with
    different sets of hyperparameters.

This guide describes two ways to build a pipeline:

*   Customize a TFX pipeline template to fit the needs of your ML workflow. TFX
    pipeline templates are prebuilt workflows that demonstrate best practices
    using the TFX standard components.
*   Build a pipeline using TFX. In this use case, you define a pipeline without
    starting from a template.

If you are new to TFX pipelines,
[learn more about the core concepts for TFX pipelines](understanding_tfx_pipelines)
before continuing.

## Overview of TFX pipelines

Note: Want to build your first pipeline before you dive into the details? Get
started
[building a pipeline using a template](#build_a_pipeline_using_a_template).

TFX pipelines are defined using the
[`Pipeline` class](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/pipeline.py){: .external }.
The following example demonstrates how to use the `Pipeline` class.

<pre class="devsite-click-to-copy prettyprint">
pipeline.Pipeline(
    pipeline_name=<var>pipeline-name</var>,
    pipeline_root=<var>pipeline-root</var>,
    components=<var>components</var>,
    enable_cache=<var>enable-cache</var>,
    metadata_connection_config=<var>metadata-connection-config</var>,
    beam_pipeline_args=<var>beam_pipeline_args</var>
)
</pre>

Replace the following:

*   <var>pipeline-name</var>: The name of this pipeline. The pipeline name must
    be unique.

    TFX uses the pipeline name when querying ML Metadata for component input
    artifacts. Reusing a pipeline name may result in unexpected behaviors.

*   <var>pipeline-root</var>: The root path of this pipeline's outputs. The root
    path must be the full path to a directory that your orchestrator has read
    and write access to. At runtime, TFX uses the pipeline root to generate
    output paths for component artifacts. This directory can be local, or on a
    supported distributed file system, such as Google Cloud Storage or HDFS.

*   <var>components</var>: A list of component instances that make up this
    pipeline's workflow.

*   <var>enable-cache</var>: (Optional.) A boolean value that indicates if this
    pipeline uses caching to speed up pipeline execution.

*   <var>metadata-connection-config</var>: (Optional.) A connection
    configuration for ML Metadata.

*   <var>beam_pipeline_args</var>: (Optional.) A set of arguments that are
    passed to the Apache Beam runner for all components that use Beam to run
    their computation.

### Defining the component execution graph

Component instances produce artifacts as outputs and typically depend on
artifacts produced by upstream component instances as inputs. The execution
sequence for component instances is determined by creating a directed acyclic
graph (DAG) of the artifact dependencies.

For instance, the `ExampleGen` standard component can ingest data from a CSV
file and output serialized example records. The `StatisticsGen` standard
component accepts these example records as input and produces dataset
statistics. In this example, the instance of `StatisticsGen` must follow
`ExampleGen` because `SchemaGen` depends on the output of `ExampleGen`.

You can also define task-based dependencies using you component's
[`add_upstream_node` and `add_downstream_node`](https://github.com/tensorflow/tfx/blob/master/tfx/components/base/base_node.py){: .external }
methods. `add_upstream_node` lets you specify that the current component must be
executed after the specified component. `add_downstream_node` lets you specify
that the current component must be executed before the specified component.

Note: Using task-based dependencies is typically not recommended. Defining the
execution graph with artifact dependencies lets you take advantage of the
automatic artifact lineage tracking and caching features of TFX.

### Caching

TFX pipeline caching lets your pipeline skip over components that have been
executed with the same set of inputs in a previous pipeline run. If caching is
enabled, the pipeline attempts to match the signature of each component, the
component and set of inputs, to one of this pipeline's previous component
executions. If there is a match, the pipeline uses the component outputs from
the previous run. If there is not a match, the component is executed.

Do not use caching if your pipeline uses non-deterministic components. For
example, if you create a component to create a random number for your pipeline,
enabling the cache causes this component to execute once. In this example,
subsequent runs use the first run's random number instead of generating a random
number.

## Build a pipeline using a template

TFX Pipeline Templates make it easier to get started with pipeline development
by providing a prebuilt pipeline that you can customize for your use case.

The following sections describe how to create a copy of a template and customize
it to meet your needs.

### Create a copy of the pipeline template

1.  Run the following command to list the TFX pipeline templates:

    <pre class="devsite-click-to-copy devsite-terminal">
    tfx template list
    </pre>

1.  Select a template from the list, currently **taxi** is the only template.
    Then run the following command:

    <pre class="devsite-click-to-copy devsite-terminal">
    tfx template copy --model=<var>template</var> --pipeline_name=<var>pipeline-name</var> \
    --destination_path=<var>destination-path</var>
    </pre>

    Replace the following:

    *   <var>template</var>: The name of the template you want to copy.
    *   <var>pipeline-name</var>: The name of the pipeline to create.
    *   <var>destination-path</var>: The path to copy the template into.

    Learn more about the [`tfx template copy` command](cli#copy).

1.  A copy of the pipeline template has been created at the path you specified.

### Explore the pipeline template

This section provides an overview of the scaffolding created by the **taxi**
template.

1.  Explore the files that were copied to your pipeline from the template. The
    **taxi** template creates the following:

    *   A **data** directory with a **data.csv** file.
    *   A **models** directory with preprocessing code and model implementations
        using `tf.estimators` and Keras.
    *   A **pipeline** directory with pipeline implementation and configuration
        scripts.
    *   The template copies the following into the destination path:

        *   DAG runner code for Apache Beam and Kubeflow Pipelines.
        *   Notebooks to explore the artifacts in the [ML Metadata](mlmd) store.

1.  Run the following command in your pipeline directory:

    <pre class="devsite-click-to-copy devsite-terminal">
    python beam_dag_runner.py
    </pre>

    The command creates a pipeline run using Apache Beam, which adds the
    following directories to your pipeline:

    *   A **tfx_metadata** directory which contains the ML Metadata store used
        locally by Apache Beam.
    *   A **tfx_pipeline_output** directory which contains the pipeline's file
        outputs.

    Note: Apache Beam is one several orchestrators which are supported in TFX.
    Apache Beam is specially suitable for running pipelines locally for faster
    iterations, possibly with smaller datasets. Apache Beam may not be suitable
    for production use since it runs on a single machine, which is more
    vulnerable to work being lost if the system becomes unavailable. TFX also
    supports orchestrators such as Apache Airflow and Kubeflow Pipeline. If
    you're using TFX with a different orchestrator, use the appropriate DAG
    runner for that orchestrator.

1.  Open your pipeline's `pipeline/configs.py` file and review the contents.
    This script defines the configuration options used by the pipeline and the
    component functions.

1.  Open your pipeline's `pipeline/pipeline.py` file and review the contents.
    This script creates the TFX pipeline. Initially, the pipeline contains only
    an ExampleGen component. Follow the instructions in the **TODO** comments in
    the pipeline files to add more steps to the pipeline.

1.  Open your pipeline's `beam_dag_runner.py` files and review the contents.
    This script creates a pipeline run and specifies the run's _parameters_,
    such as the `data_path` and `preprocessing_fn`.

1.  You have reviewed the scaffolding created by the template and created a
    pipeline run using Apache Beam. Next, customize the template to fit your
    requirements.

### Customize your pipeline

This section provides an overview of how to get started customizing the **taxi**
template.

1.  Design your pipeline. The scaffolding that a template provides helps you
    implement a pipeline for tabular data using the TFX standard components. If
    you are moving an existing ML workflow into a pipeline, you may need to
    revise your code to make full use of
    [TFX standard components](index#tfx_standard_components). You may also need
    to create [custom components](understanding_custom_components) that
    implement features which are unique to your workflow or that are not yet
    supported by TFX standard components.

1.  Once you have designed your pipeline, iteratively customize the pipeline
    using the following process. Start from the component that ingests data into
    your pipeline, which is usually the `ExampleGen` component.

    1.  Customize the pipeline or a component to fit your use case. These
        customizations may include changes like:

        *   Changing pipeline parameters.
        *   Adding components to the pipeline or removing them.
        *   Replacing the data input source. This data source can either be a
            file or queries into services such as BigQuery.
        *   Changing a component's configuration in the pipeline.
        *   Changing a component's customization function.

    1.  Run the component locally using the `beam_dag_runner.py` script, or
        another appropriate DAG runner if you are using a different
        orchestrator. If the script fails, debug the failure and retry running
        the script.

    1.  Once this customization is working, move on to the next customization.

1.  Working iteratively, you can customize each step in the template workflow to
    meet your needs.

## Build a custom pipeline

Use the following instructions to learn more about building a custom pipeline
without using a template.

1.  Design your pipeline. The TFX standard components provide proven
    functionality to help you implement a complete ML workflow. If you are
    moving an existing ML workflow into a pipeline, you may need to revise your
    code to make full use of TFX standard components. You may also need to
    create [custom components](understanding_custom_components) that implement
    features such as data augmentation.

    *   Learn more about
        [standard TFX components](index#tfx_standard_components).
    *   Learn more about [custom components](understanding_custom_components).

1.  Create a script file to define your pipeline using the following example.
    This guide refers to this file as `my_pipeline.py`.

    <pre class="devsite-click-to-copy prettyprint">
    import os
    from typing import Optional, Text, List
    from absl import logging
    from ml_metadata.proto import metadata_store_pb2
    from tfx.orchestration import metadata
    from tfx.orchestration import pipeline
    from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

    PIPELINE_NAME = 'my_pipeline'
    PIPELINE_ROOT = os.path.join('.', 'my_pipeline_output')
    METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
    ENABLE_CACHE = True

    def create_pipeline(
      pipeline_name: Text,
      pipeline_root:Text,
      enable_cache: bool,
      metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
      beam_pipeline_args: Optional[List[Text]] = None
    ):
      components = []

      return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args,
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
          )

      BeamDagRunner().run(my_pipeline)

    if __name__ == '__main__':
      logging.set_verbosity(logging.INFO)
      run_pipeline()
    </pre>

    In the coming steps, you define your pipeline in `create_pipeline` and run
    your pipeline locally using Apache Beam in `run_pipeline`.

    Iteratively build your pipeline using the following process.

    1.  Customize the pipeline or a component to fit your use case. These
        customizations may include changes like:

        *   Changing pipeline parameters.
        *   Adding components to the pipeline or removing them.
        *   Replacing a data input file.
        *   Changing a component's configuration in the pipeline.
        *   Changing a component's customization function.

    1.  Run the component locally using Apache Beam or another orchestrator by
        running the script file. If the script fails, debug the failure and
        retry running the script.

    1.  Once this customization is working, move on to the next customization.

    Start from the first node in your pipeline's workflow, typically the first
    node ingests data into your pipeline.

1.  Add the first node in your workflow to your pipeline. In this example, the
    pipeline uses the `ExampleGen` standard component to load a CSV from a
    directory at `./data`.

    <pre class="devsite-click-to-copy prettyprint">
    from tfx.components import CsvExampleGen
    from tfx.utils.dsl_utils import external_input

    DATA_PATH = os.path.join('.', 'data')

    def create_pipeline(
      pipeline_name: Text,
      pipeline_root:Text,
      data_path: Text,
      enable_cache: bool,
      metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
      beam_pipeline_args: Optional[List[Text]] = None
    ):
      components = []

      example_gen = CsvExampleGen(input=external_input(data_path))
      components.append(example_gen)

      return pipeline.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args,
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA_PATH)
          )

      BeamDagRunner().run(my_pipeline)
    </pre>

    `CsvExampleGen` creates serialized example records using the data in the CSV
    at the specified data path. By setting the `CsvExampleGen` component's
    `input` parameter with
    [`external_input`](https://github.com/tensorflow/tfx/blob/master/tfx/utils/dsl_utils.py){: .external },
    you specify that the data path is passed into the pipeline and that the path
    should be stored as an artifact.

1.  Create a `data` directory in the same directory as `my_pipeline.py`. Add a
    small CSV file to the `data` directory.

1.  Use the following command to run your `my_pipeline.py` script and test the
    pipeline with Apache Beam or another orchestrator.

    <pre class="devsite-click-to-copy devsite-terminal">
    python my_pipeline.py
    </pre>

    The result should be something like the following:

    <pre>
    INFO:absl:Component CsvExampleGen depends on [].
    INFO:absl:Component CsvExampleGen is scheduled.
    INFO:absl:Component CsvExampleGen is running.
    INFO:absl:Running driver for CsvExampleGen
    INFO:absl:MetadataStore with DB connection initialized
    INFO:absl:Running executor for CsvExampleGen
    INFO:absl:Generating examples.
    INFO:absl:Using 1 process(es) for Beam pipeline execution.
    INFO:absl:Processing input csv data ./data/* to TFExample.
    WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
    INFO:absl:Examples generated.
    INFO:absl:Running publisher for CsvExampleGen
    INFO:absl:MetadataStore with DB connection initialized
    INFO:absl:Component CsvExampleGen is finished.
    </pre>

1.  Continue to iteratively add components to your pipeline.
