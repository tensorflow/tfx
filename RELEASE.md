# Current version (not yet released; still in development)

## Major Features and Improvements

*   Made model validator executor forward compatible with TFMA change.

## Bug fixes and other changes

### Deprecations

## Breaking changes

### For pipeline authors

### For component authors

## Documentation updates

*   Added conceptual info on Artifacts to guide/index.md


# Version 0.14.0

## Major Features and Improvements

*   Added support for Google Cloud ML Engine Training and Serving as extension.
*   Supported pre-split input for ExampleGen components
*   Added ImportExampleGen component for importing tfrecord files with TF
    Example data format
*   Added a generic ExampleGen component to reduce the work of custom ExampleGen
*   Released Python 3 type hints and added support for Python 3.6 and 3.7.
*   Added an Airflow integration test for chicago_taxi_simple example.
*   Updated tfx docker image to use Python 3.6 on Ubuntu 16.04.
*   Added example for how to define and add a custom component.
*   Added PrestoExampleGen component.
*   Added Parquet executor for ExampleGen component.
*   Added Avro executor for ExampleGen component.
*   Enables Kubeflow Pipelines users to specify arbitrary ContainerOp decorators
    that can be applied to each pipeline step.
*   Added scripts and instructions for running the TFX Chicago Taxi example on
    Spark (via Apache Beam).
*   Introduced a new mechanism of artifact info passing between components that
    relies solely on ML Metadata.
*   Unified driver and execution logging to go through tf.logging.
*   Added support for Beam as an orchestrator.
*   Introduced the experimental InteractiveContext environment for iterative
    notebook development, as well as an example Chicago Taxi notebook in this
    environment with TFDV / TFMA examples.
*   Enabled Transform and Trainer components to specify user defined function
    (UDF) module by Python module path in addition to path to a module file.
*   Enable ImportExampleGen component for Kubeflow.
*   Enabled SchemaGen to infer feature shape.
*   Enabled metadata logging and pipeline caching capability for KubeflowRunner.
*   Used custom container for AI Platform Trainer extension.
*   Introduced ExecutorSpec, which generalizes the representation of executors
    to include both Python classes and containers.
*   Supported run context for metadata tracking of tfx pipeline.


### Deprecations
*   Deprecated 'metadata_db_root' in favor of passing in
    metadata_connection_config directly.
*   airflow_runner.AirflowDAGRunner is renamed to
    airflow_dag_runner.AirflowDagRunner.
*   runner.KubeflowRunner is renamed to kubeflow_dag_runner.KubeflowDagRunner.
*   The "input" and "output" exec_properties fields for ExampleGen executors
    have been renamed to "input_config" and "output_config", respectively.
*   Declared 'cmle_training_args' on trainer and 'cmle_serving_args' on pusher
    deprecated. User should use the `trainer/pusher` executors in
    tfx.extensions.google_cloud_ai_platform module instead.
*   Moved tfx.orchestration.gcp.cmle_runner to
    tfx.extensions.google_cloud_ai_platform.runner.
*   Deprecated csv_input and tfrecord_input, use external_input instead.

## Bug fixes and other changes
*   Updated components and code samples to use `tft.TFTransformOutput` (
    introduced in tensorflow_transform 0.8). This avoids directly accessing the
    DatasetSchema object which may be removed in tensorflow_transform 0.14 or
    0.15.
*   Fixed issue #113 to have consistent type of train_files and eval_files
    passed to trainer user module.
*   Fixed issue #185 preventing the Airflow UI from visualizing the component's
    subdag operators and logs.
*   Fixed issue #201 to make GCP credentials optional.
*   Bumped dependency to kfp (Kubeflow Pipelines SDK) to be at version at least
    0.1.18.
*   Updated code example to
    *   use 'tf.data.TFRecordDataset' instead of the deprecated function
        'tf.TFRecordReader'
    *   add test to train, evaluate and export.
*   Component definition streamlined with explicit ComponentSpec and new style
    for defining component classes.
*   TFX now depends on `pyarrow>=0.14.0,<0.15.0` (through its dependency on
    `tensorflow-data-validation`).
*   Introduced 'examples' to the Trainer component API. It's recommended to use
    this field instead of 'transformed_examples' going forward.
*   Trainer can now run without the 'transform_output' input.
*   Added check for duplicated component ids within a pipeline.
*   String representations for Channel and Artifact (TfxType) classes were
    improved.
*   Updated workshop/setup/setup_demo.sh to fix version incompatibilities
*   Updated workshop by adding note and instructions to fix issue with GCC
    version when starting `airflow webserver`.
*   Prepared support for analyzer cache optimization in transform executor.
*   Fixed issue #463 correcting syntax in SCHEMA_EMPTY message.
*   Added an explicit check that pipeline name cannot exceed 63 characters.
*   SchemaGen takes a new argument, infer_feature_shape to indicate whether to
    infer shape of features in schema. Current default value is False, but we
    plan to remove default value for it in future.
*   Depended on 'click>=7.0,<8'
*   Depended on `apache-beam[gcp]>=2.14,<3`
*   Depended on `ml-metadata>=-1.14.0,<0.15`
*   Depended on `tensorflow-data-validation>=0.14.1,<0.15`
*   Depended on `tensorflow-model-analysis>=0.14.0,<0.15`
*   Depended on `tensorflow-transform>=0.14.0,<0.15`

## Breaking changes

### For pipeline authors

*   The "outputs" argument, which is used to override the automatically-
    generated output Channels for each component class has been removed; the
    equivalent overriding functionality is now available by specifying optional
    keyword arguments (see each component class definition for details).
*   The optional arguments "executor" and "unique_name" of component classes
    have been uniformly renamed to "executor_spec" and "instance_name",
    respectively.
*   The "driver" optional argument of component classes is no longer available:
    users who need to override the driver for a component should subclass the
    component and override the DRIVER_CLASS field.
*   The `example_gen.component.ExampleGen` class has been refactored into the
    `example_gen.component._QueryBasedExampleGen` and
    `example_gen.component.FileBasedExampleGen` classes.
*   pipeline_root passed to pipeline.Pipeline is now the root to the running
    pipeline instead of root of all pipelines.

### For component authors

*   Component class definitions have been simplified; existing custom components
    need to:
    * specify a ComponentSpec contract and conform to new class definition
      style (see `base_component.BaseComponent`)
    * specify `EXECUTOR_SPEC=ExecutorClassSpec(MyExecutor)` in the component
      definition to replace `executor=MyExecutor` from component constructor.
*   Artifact definitions for standard TFX components have moved from using
    string type names into being concrete Artifact classes (see each official
    TFX component's ComponentSpec definition in `types.standard_component_specs`
    and the definition of built-in Artifact types in
    `types.standard_artifacts`).
*   The `base_component.ComponentOutputs` class has been renamed to
    `base_component._PropertyDictWrapper`.
*   The tfx.utils.types.TfxType class has been renamed to tfx.types.Artifact.
*   The tfx.utils.channel.Channel class has been moved to tfx.types.Channel.
*   The "static_artifact_collection" argument to types.Channel has been renamed
    to "artifacts".
*   ArtifactType for artifacts will have two new properties: pipeline_name and
    producer_component.
*   The ARTIFACT_STATE_* constants were consolidated into the
    types.artifacts.ArtifactState enum class.

# Version 0.13.0

## Major Features and Improvements

*   Adds support for Python 3.5
*   Initial version of following orchestration platform supported:
    *   Kubeflow
*   Added TensorFlow Model Analysis Colab example
*   Supported split ratio for ExampleGen components
*   Supported running a single executor independently

## Bug fixes and other changes

*   Fixes issue #43 that prevent new execution in some scenarios
*   Fixes issue #47 that causes ImportError on chicago_taxi execution on
    dataflow
*   Depends on `apache-beam[gcp]>=2.12,<3`
*   Depends on `tensorflow-data-validation>=0.13.1,<0.14`
*   Depends on `tensorflow-model-analysis>=0.13.2,<0.14`
*   Depends on `tensorflow-transform>=0.13,<0.14`
*   Deprecations:
    *   PipelineDecorator is deprecated. Please construct a pipeline directly
        from a list of components instead.
*   Increased verbosity of logging to container stdout when running under
    Kubeflow Pipelines.
*   Updated developer tutorial to support Python 3.5+

## Breaking changes

*   Examples code are moved from 'examples' to 'tfx/examples': this ensures that
    PyPi package contains only one top level python module 'tfx'.

## Things to notice for upgrading

*   Multiprocessing on Mac OS >= 10.13 might crash for Airflow. See
    [AIRFLOW-3326](https://issues.apache.org/jira/browse/AIRFLOW-3326) for
    details and solution.

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

