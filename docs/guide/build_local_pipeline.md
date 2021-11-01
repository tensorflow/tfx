# Building a TFX Pipeline Locally

TFX makes it easier to orchestrate your machine learning (ML) workflow as a
pipeline, in order to:

*   Automate your ML process, which lets you regularly retrain, evaluate, and
    deploy your model.
*   Create ML pipelines which include deep analysis of model performance and
    validation of newly trained models to ensure performance and reliability.
*   Monitor training data for anomalies and eliminate training-serving skew
*   Increase the velocity of experimentation by running a pipeline with
    different sets of hyperparameters.

A typical pipeline development process begins on a local machine, with data
analysis and component setup, before being deployed into production. This guide
describes two ways to build a pipeline locally.

*   Customize a TFX pipeline template to fit the needs of your ML workflow. TFX
    pipeline templates are prebuilt workflows that demonstrate best practices
    using the TFX standard components.
*   Build a pipeline using TFX. In this use case, you define a pipeline without
    starting from a template.

As you are developing your pipeline, you can run it with `LocalDagRunner`.
Then, once the pipeline components have been well defined and tested, you would
use a production-grade orchestrator such as Kubeflow or Airflow.

## Before you begin

TFX is a Python package, so you will need to set up a Python development
environment, such as a virtual environment or a Docker container. Then:

```bash
pip install tfx
```

If you are new to TFX pipelines,
[learn more about the core concepts for TFX pipelines](understanding_tfx_pipelines)
before continuing.

## Build a pipeline using a template

TFX Pipeline Templates make it easier to get started with pipeline development
by providing a prebuilt set of pipeline definitions that you can customize for
your use case.

The following sections describe how to create a copy of a template and customize
it to meet your needs.

### Create a copy of the pipeline template

1.  See list of the available TFX pipeline templates:

    <pre class="devsite-click-to-copy devsite-terminal">
    tfx template list
    </pre>

1.  Select a template from the list

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

Note: The rest of this guide assumes you selected the `penguin` template.

### Explore the pipeline template

This section provides an overview of the scaffolding created by a template.

1.  Explore the directories and files that were copied to your pipeline's root
    directory

    *   A **pipeline** directory with
        -   `pipeline.py` - defines the pipeline, and lists which components are
            being used
        -   `configs.py` - hold configuration details such as where the data is
            coming from or which orchestrator is being used
    *   A **data** directory
        -   This typically contains a `data.csv` file, which is the default
            source for `ExampleGen`. You can change the data source in
            `configs.py`.
    *   A **models** directory with preprocessing code and model implementations

    *   The template copies DAG runners for local environment and Kubeflow.

    *   Some templates also include Python Notebooks so that you can explore
        your data and artifacts with Machine Learning MetaData.

1.  Run the following commands in your pipeline directory:

    <pre class="devsite-click-to-copy devsite-terminal">
    tfx pipeline create --pipeline_path local_runner.py
    </pre>

    <pre class="devsite-click-to-copy devsite-terminal">
    tfx run create --pipeline_name <var>pipeline_name</var>
    </pre>

    The command creates a pipeline run using `LocalDagRunner`, which adds the
    following directories to your pipeline:

    *   A **tfx_metadata** directory which contains the ML Metadata store used
        locally.
    *   A **tfx_pipeline_output** directory which contains the pipeline's file
        outputs.

    Note: `LocalDagRunner` is one several orchestrators which are supported in
    TFX. It is specially suitable for running pipelines locally for faster
    iterations, possibly with smaller datasets. `LocalDagRunner` may not be
    suitable for production use since it runs on a single machine, which is more
    vulnerable to work being lost if the system becomes unavailable. TFX also
    supports orchestrators such as Apache Beam, Apache Airflow, and Kubeflow
    Pipeline. If you're using TFX with a different orchestrator, use the
    appropriate DAG runner for that orchestrator.

    Note: As of this writing, `LocalDagRunner` is used in the `penguin`
    template, while the `taxi` template uses Apache Beam. The config files for
    the `taxi` template are set up to use Beam, and the CLI command is the same.

1.  Open your pipeline's `pipeline/configs.py` file and review the contents.
    This script defines the configuration options used by the pipeline and the
    component functions. This is where you would specify things like the
    location of the datasource or the number of training steps in a run.

1.  Open your pipeline's `pipeline/pipeline.py` file and review the contents.
    This script creates the TFX pipeline. Initially, the pipeline contains only
    an `ExampleGen` component.

    -   Follow the instructions in the **TODO** comments in `pipeline.py` to add
        more steps to the pipeline.

1.  Open `local_runner.py` file and review the contents. This script creates a
    pipeline run and specifies the run's _parameters_, such as the `data_path`
    and `preprocessing_fn`.

1.  You have reviewed the scaffolding created by the template and created a
    pipeline run using `LocalDagRunner`. Next, customize the template to fit
    your requirements.

### Customize your pipeline

This section provides an overview of how to get started customizing your
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

    1.  Run the component locally using the `local_runner.py` script, or another
        appropriate DAG runner if you are using a different orchestrator. If the
        script fails, debug the failure and retry running the script.

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
    import tfx.v1 as tfx

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

      return tfx.dsl.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args, <!-- needed? -->
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          enable_cache=ENABLE_CACHE,
          metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
          )

      tfx.orchestration.LocalDagRunner().run(my_pipeline)

    if __name__ == '__main__':
      logging.set_verbosity(logging.INFO)
      run_pipeline()
    </pre>

    In the coming steps, you define your pipeline in `create_pipeline` and run
    your pipeline locally using the local runner.

    Iteratively build your pipeline using the following process.

    1.  Customize the pipeline or a component to fit your use case. These
        customizations may include changes like:

        *   Changing pipeline parameters.
        *   Adding components to the pipeline or removing them.
        *   Replacing a data input file.
        *   Changing a component's configuration in the pipeline.
        *   Changing a component's customization function.

    1.  Run the component locally using the local runner or by running the
        script directly. If the script fails, debug the failure and retry
        running the script.

    1.  Once this customization is working, move on to the next customization.

    Start from the first node in your pipeline's workflow, typically the first
    node ingests data into your pipeline.

1.  Add the first node in your workflow to your pipeline. In this example, the
    pipeline uses the `ExampleGen` standard component to load a CSV from a
    directory at `./data`.

    <pre class="devsite-click-to-copy prettyprint">
    from tfx.components import CsvExampleGen

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

      example_gen = tfx.components.CsvExampleGen(input_base=data_path)
      components.append(example_gen)

      return tfx.dsl.Pipeline(
            pipeline_name=pipeline_name,
            pipeline_root=pipeline_root,
            components=components,
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
            beam_pipeline_args=beam_pipeline_args, <!-- needed? -->
        )

    def run_pipeline():
      my_pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_path=DATA_PATH,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
        )

      tfx.orchestration.LocalDagRunner().run(my_pipeline)
    </pre>

    `CsvExampleGen` creates serialized example records using the data in the CSV
    at the specified data path. By setting the `CsvExampleGen` component's
    `input_base` parameter with the data root.

1.  Create a `data` directory in the same directory as `my_pipeline.py`. Add a
    small CSV file to the `data` directory.

1.  Use the following command to run your `my_pipeline.py` script.

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
    INFO:absl:Using 1 process(es) for Local pipeline execution.
    INFO:absl:Processing input csv data ./data/* to TFExample.
    WARNING:root:Couldn't find python-snappy so the implementation of _TFRecordUtil._masked_crc32c is not as fast as it could be.
    INFO:absl:Examples generated.
    INFO:absl:Running publisher for CsvExampleGen
    INFO:absl:MetadataStore with DB connection initialized
    INFO:absl:Component CsvExampleGen is finished.
    </pre>

1.  Continue to iteratively add components to your pipeline.
