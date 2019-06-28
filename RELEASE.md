# Current version (not yet released; still in development)

## Major Features and Improvements
*   Added support for Google Cloud ML Engine Training and Serving as extension.
*   Supported pre-split input for ExampleGen components
*   Added ImportExampleGen component for importing tfrecord files with
    TF Example data format
*   Added a generic ExampleGen component to reduce the work of custom ExampleGen
*   Released Python 3 type hints.
*   Added an Airflow integration test for chicago_taxi_simple example.
*   Updated tfx docker image to use Python 3.
*   Added example for how to define and add a custom component.
*   Added Parquet executor for ExampleGen component.
*   Enables Kubeflow Pipelines users to specify arbitrary ContainerOp decorators
    that can be applied to each pipeline step.
*   Added scripts and instructions to run on Spark via Beam to the Chicago Taxi
    examples
*   Introduced new mechanism of artifact info passing between component that
    solely rely on ML Metadata

## Bug fixes and other changes
*   Declared 'cmle_training_args' on trainer and 'cmle_serving_args' on
    pusher deprecated. User should use the `trainer/pusher` executors in
    tfx.extensions.google_cloud_ai_platform module instead.
*   Update components and code samples to use `tft.TFTransformOutput` (
    introduced in tensorflow_transform 0.8).  This avoids directly accessing the
    DatasetSchema object which may be removed in tensorflow_transform 0.14 or
    0.15.
*   Fixes issue #113 to have consistent type of train_files and eval_files
    passed to trainer user module.
*   TfxType has been renamed to TfxArtifact.
*   Fixes issue #185 preventing the Airflow UI from visualizing the component's
    subdag operators and logs.
*   Fixes issue #201 so GCP credentials are optional.
*   Bumped dependency to kfp (Kubeflow Pipelines SDK) version to be at later
    than 0.1.18.
*   Updated code example to
    * use 'tf.data.TFRecordDataset' instead of the deprecated function
      'tf.TFRecordReader'
    * add test to train, evaluate and export.
*   Component definition streamlined with explicit ComponentSpec and new style
    for defining component classes.
*   Moved tfx.orchestration.gcp.cmle_runner to
    tfx.extensions.google_cloud_ai_platform.runner.
*   Depends on `pyarrow>=0.11.1,<0.12.0`

## Breaking changes
*   Component class definitions have been simplified; existing custom components
    need to specify a ComponentSpec contract and conform to new class definition
    style (see `base_component.BaseComponent`).
*   The "outputs" argument, which is used to override the automatically-
    generated output Channels for each component class has been removed; the
    equivalent overriding functionality is now available by specifying
    optional keyword arguments (see each component class definition for
    details).
*   The optional arguments "executor" and "unique_name" of component classes
    have been uniformly renamed to "executor_class" and "name", respectively.
    The "driver" optional argument of component classes is no longer available:
    users who need to override the driver for a component should subclass the
    component and override the DRIVER_CLASS field.
*   The `example_gen.component.ExampleGen` class has been refactored into the
    `example_gen.component._ExampleGen` and
    `example_gen.component._FileBasedExampleGen` abstract classes. Users should
    use their concrete subclasses instead of using these abstract components
    directly.
*   The "input" and "output" exec_properties fields for ExampleGen executors
    have been renamed to "input_config" and "output_config", respectively.
*   The `base_component.ComponentOutputs` class has been renamed to
    `base_component._PropertyDictWrapper`.
*   The utils.types.TfxType class has been renamed to utils.types.TfxArtifact.
*   The "static_artifact_collection" argument to utils.channel.Channel has been
    renamed to "artifacts".
*   ArtifactType for artifacts will have two new properties: pipeline_name and
    producer_component.
*   The recommended method of accessing an input/output of a ComponentSpec has
    changed from `spec.outputs.output_name` to `spec.outputs['output_name']`.
    The previous style will be deprecated soon.

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
*   Fixes issue #47 that causes ImportError on chicago_taxi execution on dataflow
*   Depends on `apache-beam[gcp]>=2.12,<3`
*   Depends on `tensorflow-data-validation>=0.13.1,<0.14`
*   Depends on `tensorflow-model-analysis>=0.13.2,<0.14`
*   Depends on `tensorflow-transform>=0.13,<0.14`
*   Deprecations:
    *    PipelineDecorator is deprecated. Please construct a pipeline directly from a list of components instead.
*   Increased verbosity of logging to container stdout when running under
    Kubeflow Pipelines.
*   Updated developer tutorial to support Python 3.5+

## Breaking changes
*   Examples code are moved from 'examples' to 'tfx/examples': this ensures that PyPi package contains only one top level python module 'tfx'.

## Things to notice for upgrading
*   Multiprocessing on Mac OS >= 10.13 might crash for Airflow. See
    [AIRFLOW-3326](https://issues.apache.org/jira/browse/AIRFLOW-3326)
    for details and solution.

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
