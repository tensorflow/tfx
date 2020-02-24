# Current version (not yet released; still in development)

## Major Features and Improvements
*   Pipelines compiled using KubeflowDagRunner now defaults to using the
    gRPC-based MLMD server deployed in Kubeflow Pipelines clusters when
    performing operations on pipeline metadata.
*   Added tfx model rewriting and tflite rewriter.
*   Added LatestBlessedModelResolver as an experimental feature which gets the
    latest model that was blessed by model validator.
*   The specific `Artifact` subclass that was serialized (if defined in the
    deserializing environment) will be used when deserializing `Artifact`s and
    when reading `Artifact`s from ML Metadata (previously, objects of the
    generic `tfx.types.artifact.Artifact` class were created in some cases).
*   Updated Evaluator's executor to support model validation.
*   Introduced awareness of chief worker to Trainer's executor, in case running
    in distributed training cluster.

## Bug fixes and other changes
*   Added --skaffold_cmd flag when updating a pipeline for kubeflow in CLI.
*   Changed python_version to 3.7 when using TF 1.15 and later for Cloud AI Platform Prediction.
*   Added 'tfx_runner' label for CAIP, BQML and Dataflow jobs submitted from
    TFX components.

### Deprecations

## Breaking changes
*   Remove "NOT_BLESSED" artifact.
*   Change constants ARTIFACT_PROPERTY_BLESSED_MODEL_* to ARTIFACT_PROPERTY_BASELINE_MODEL_*.

### For pipeline authors

### For component authors

## Documentation updates

# Version 0.21.0

## Major Features and Improvements

*   TFX version 0.21.0 will be the last version of TFX supporting Python 2.
*   Added support for `RuntimeParameter`s to allow users can specify templated
    values at runtime. This is currently only supported in Kubeflow Pipelines.
    Currently, only attributes in `ComponentSpec.PARAMETERS` and the URI of
    external artifacts can be parameterized (component inputs / outputs can
    not yet be parameterized). See
    `tfx/examples/chicago_taxi_pipeline/taxi_pipeline_runtime_parameter.py`
    for example usage.
*   Users can access the parameterized pipeline root when defining the
    pipeline by using the `pipeline.ROOT_PARAMETER` placeholder in
    KubeflowDagRunner.
*   Users can pass appropriately encoded Python `dict` objects to specify
    protobuf parameters in `ComponentSpec.PARAMETERS`; these will be decoded
    into the proper protobuf type. Users can avoid manually constructing complex
    nested protobuf messages in the component interface.
*   Added support in Trainer for using other model artifacts. This enables
    scenarios such as warm-starting.
*   Updated trainer executor to pass through custom config to the user module.
*   Artifact type-specific properties can be defined through overriding the
    `PROPERTIES` dictionary of a `types.artifact.Artifact` subclass.
*   Added new example of chicago_taxi_pipeline on Google Cloud Bigquery ML.
*   Added support for multi-core processing in the Flink and Spark Chicago Taxi
    PortableRunner example.
*   Added a metadata adapter in Kubeflow to support logging the Argo pod ID as
    an execution property.
*   Added a prototype Tuner component and an end-to-end iris example.
*   Created new generic trainer executor for non estimator based model, e.g.,
    native Keras.
*   Updated to support passing `tfma.EvalConfig` in evaluator when calling TFMA.
*   Users can create a pipeline using a new experimental CLI command,
    `template`.
*   Added an iris example with native Keras.

## Bug fixes and other changes
*   Switched the default behavior of KubeflowDagRunner to not mounting GCP
    secret.
*   Fixed "invalid spec: spec.arguments.parameters[6].name 'pipeline-root' is
    not unique" error when the user include `pipeline.ROOT_PARAMETER` and run
    pipeline on KFP.
*   Added support for an hparams artifact as an input to Trainer in
    preparation for tuner support.
*   Refactored common dependencies in the TFX dockerfile to a base image to
    improve the reliability of image building process.
*   Fixes missing Tensorboard link in KubeflowDagRunner.
*   Depends on `apache-beam[gcp]>=2.17,<2.18`
*   Depends on `ml-metadata>=0.21,<0.22`.
*   Depends on `tensorflow-data-validation>=0.21,<0.22`.
*   Depends on `tensorflow-model-analysis>=0.21,<0.22`.
*   Depends on `tensorflow-transform>=0.21,<0.22`.
*   Depends on `tfx-bsl>=0.21,<0.22`.
*   Depends on `pyarrow>=0.14,<0.15`.
*   Removed `tf.compat.v1` usage for iris and cifar10 examples.
*   CSVExampleGen: started using the CSV decoding utilities in `tfx-bsl`
    (`tfx-bsl>=0.15.2`)
*   Fixed problems with Airflow tutorial notebooks.
*   Added performance improvements for the Transform Component (for statistics
    generation).
*   Raised exceptions when container building fails.
*   Enhanced custom slack component by adding a kubeflow example.
*   Allowed windows style paths in Transform component cache.
*   Fixed bug in CLI (--engine=kubeflow) which uses hard coded obsolete image
    (TFX 0.14.0) as the base image.
*   Fixed bug in CLI (--engine=kubeflow) which could not handle skaffold
    response when an already built image is reused.
*   Allowed users to specify the region to use when serving with AI Platform.
*   Allowed users to give deterministic job id to AI Platform Training job.
*   System-managed artifact properties ("name", "state", "pipeline_name" and
    "producer_component") are now stored as ML Metadata artifact custom
    properties.
*   Fixed loading trainer and transformation functions from python module files
    without the .py extension.
*   Fixed some ill-formed visualization when running on KFP.
*   Removed system info from artifact properties and use channels to hold info
    for generating MLMD queries.
*   Rely on MLMD context for inter-component artifact resolution and execution
    publishing.
*   Added pipeline level context and component run level context.
*   Included test data for examples/chicago_taxi_pipeline in package.
*   Changed `BaseComponentLauncher` to require the user to pass in an ML
    Metadata connection object instead of a ML Metadata connection config.
*   Capped version of Tensorflow runtime used in Google Cloud integration to
    1.15.
*   Updated Chicago Taxi example dependencies to Beam 2.17.0, Flink 1.9.1, Spark
    2.4.4.
*   Fixed an issue where `build_ephemeral_package()` used an incorrect path to
    locate the `tfx` directory.
*   The ImporterNode now allows specification of general artifact properties.
*   Added 'tfx_executor', 'tfx_version' and 'tfx_py_version' labels for CAIP,
    BQML and Dataflow jobs submitted from TFX components.

### Deprecations

## Breaking changes

### For pipeline authors
*   Standard artifact TYPE_NAME strings were reconciled to match their class
    names in `types.standard_artifacts`.
*   The "split" property on multiple artifacts has been replaced with the
    JSON-encoded "split_names" property on a single grouped artifact.
*   The execution caching mechanism was changed to rely on ML Metadata
    pipeline context. Existing cached executions will not be reused when running
    on this version of TFX for the first time.
*   The "split" property on multiple artifacts has been replaced with the
    JSON-encoded "split_names" property on a single grouped artifact.

### For component authors
*   Artifact type name strings to the `types.artifact.Artifact` and
    `types.channel.Channel` classes are no longer supported; usage here should
    be replaced with references to the artifact subclasses defined in
    `types.standard_artfacts.*` or to custom subclasses of
    `types.artifact.Artifact`.

## Documentation updates

# Version 0.15.0

## Major Features and Improvements

*   Offered unified CLI for tfx pipeline actions on various orchestrators
    including Apache Airflow, Apache Beam and Kubeflow.
*   Polished experimental interactive notebook execution and visualizations so
    they are ready for use.
*   Added BulkInferrer component to TFX pipeline, and corresponding offline
    inference taxi pipeline.
*   Introduced ImporterNode as a special TFX node to register external resource
    into MLMD so that downstream nodes can use as input artifacts. An example
    `taxi_pipeline_importer.py` enabled by ImporterNode was added to showcase
    the user journey of user-provided schema (issue #571).
*   Added experimental support for TFMA fairness indicator thresholds.
*   Demonstrated DirectRunner multi-core processing in Chicago Taxi example,
    including Airflow and Beam.
*   Introduced `PipelineConfig` and `BaseComponentConfig` to control the
    platform specific settings for pipelines and components.
*   Added a custom Executor of Pusher to push model to BigQuery ML for serving.
*   Added KubernetesComponentLauncher to support launch ExecutorContainerSpec in
    a Kubernetes cluster.
*   Made model validator executor forward compatible with TFMA change.
*   Added Iris flowers classification example.
*   Added support for serialization and deserialization of components.
*   Made component launcher extensible to support launching components on
    multiple platforms.
*   Simplified component package names.
*   Introduced BaseNode as the base class of any node in a TFX pipeline DAG.
*   Added docker component launcher to launch container component.
*   Added support for specifying pipeline root in runtime when run on
    KubeflowDagRunner. A default value can be provided when constructing the TFX
    pipeline.
*   Added basic span support in ExampleGen to ingest file based data sources
    that can be updated regularly by upstream.
*   Branched serving examples under chicago_taxi_pipeline/ from chicago_taxi/
    example.
*   Supported beam arg 'direct_num_workers' for multi-processing on local.
*   Improved naming of standard component inputs and outputs.
*   Improved visualization functionality in the experimental TFX notebook
    interface.
*   Allowed users to specify output file format when compiling TFX pipelines
    using KubeflowDagRunner.
*   Introduced ResolverNode as a special TFX node to resolve input artifacts for
    downstream nodes. ResolverNode is a convenient way to wrap TFX Resolver, a
    logical unit for resolving input artifacts.
*   Added cifar-10 example to demonstrate image classification.
*   Added container builder feature in the CLI tool for container-based custom
    python components. This is specifically for the Kubeflow orchestration
    engine, which requires containers built with the custom python code.
*   Demonstrated DirectRunner multi-core processing in Chicago Taxi example,
    including Airflow and Beam.
*   Added Kubeflow artifact visualization of inputs, outputs and execution
    properties for components using a Markdown file. Added Tensorboard to
    Trainer components as well.

## Bug fixes and other changes

*   Bumped test dependency to kfp (Kubeflow Pipelines SDK) to be at version
    0.1.31.2.
*   Fixed trainer executor to correctly make `transform_output` optional.
*   Updated Chicago Taxi example dependency tensorflow to version >=1.14.0.
*   Updated Chicago Taxi example dependencies tensorflow-data-validation,
    tensorflow-metadata, tensorflow-model-analysis, tensorflow-serving-api, and
    tensorflow-transform to version >=0.14.
*   Updated Chicago Taxi example dependencies to Beam 2.14.0, Flink 1.8.1, Spark
    2.4.3.
*   Adopted new recommended way to access component inputs/outputs as
    `component.outputs['output_name']` (previously, the syntax was
    `component.outputs.output_name`).
*   Updated Iris example to skip transform and use Keras model.
*   Fixed the check for input artifact existence in base driver.
*   Fixed bug in AI Platform Pusher that prevents pushes after first model, and
    not being marked as default.
*   Replaced all usage of deprecated `tensorflow.logging` with `absl.logging`.
*   Used special user agent for all HTTP requests through googleapiclient and
    apitools.
*   Transform component updated to use `tf.compat.v1` according to the TF 2.0
    upgrading procedure.
*   TFX updated to use `tf.compat.v1` according to the TF 2.0 upgrading
    procedure.
*   Added Kubeflow local example pipeline that executes components in-cluster.
*   Fixed a bug that prevents updating execution type.
*   Fixed a bug in model validator driver that reads across pipeline boundaries
    when resolving latest blessed model.
*   Depended on `apache-beam[gcp]>=2.16,<3`
*   Depended on `ml-metadata>=0.15,<0.16`
*   Depended on `tensorflow>=1.15,<3`
*   Depended on `tensorflow-data-validation>=0.15,<0.16`
*   Depended on `tensorflow-model-analysis>=0.15.2,<0.16`
*   Depended on `tensorflow-transform>=0.15,<0.16`
*   Depended on 'tfx_bsl>=0.15.1,<0.16'
*   Made launcher return execution information, containing populated inputs,
    outputs, and execution id.
*   Updated the default configuration for accessing MLMD from pipelines running
    in Kubeflow.
*   Updated Airflow developer tutorial
*   CSVExampleGen: started using the CSV decoding utilities in `tfx-bsl`
    (`tfx-bsl>=0.15.2`)
*   Added documentation for Fairness Indicators.

### Deprecations

*   Deprecated component_type in favor of type.
*   Deprecated component_id in favor of id.
*   Move beam_pipeline_args out of additional_pipeline_args as top level
    pipeline param
*   Deprecated chicago_taxi folder, beam setup scripts and serving examples are
    moved to chicago_taxi_pipeline folder.

## Breaking changes

*   Moved beam setup scripts from examples/chicago_taxi/ to
    examples/chicago_taxi_pipeline/
*   Moved interactive notebook classes into `tfx.orchestration.experimental`
    namespace.
*   Starting from 1.15, package `tensorflow` comes with GPU support. Users won't
    need to choose between `tensorflow` and `tensorflow-gpu`. If any GPU devices
    are available, processes spawned by all TFX components will try to utilize
    them; note that in rare cases, this may exhaust the memory of the device(s).
*   Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU support. If
    `tensorflow-gpu` 2.0.0 is installed before installing `tfx`, it will be
    replaced with `tensorflow` 2.0.0. Re-install `tensorflow-gpu` 2.0.0 if
    needed.
*   Caveat: MLMD schema auto-upgrade is now disabled by default. For users who
    upgrades from 0.13 and do not want to lose the data in MLMD, please refer to
    [MLMD documentation](https://github.com/google/ml-metadata/blob/master/g3doc/get_started.md#upgrade-mlmd-library)
    for guide to upgrade or downgrade MLMD database. Users who upgraded from TFX
    0.14 should not be affected since there is not schema change between these
    two versions.

### For pipeline authors

*   Deprecated the usage of `tf.contrib.training.HParams` in Trainer as it is
    deprecated in TF 2.0. User module relying on member method of that class
    will not be supported. Dot style property access will be the only supported
    style from now on.
*   Any SavedModel produced by tf.Transform <=0.14 using any tf.contrib ops (or
    tf.Transform ops that used tf.contrib ops such as tft.quantiles,
    tft.bucketize, etc.) cannot be loaded with TF 2.0 since the contrib library
    has been removed in 2.0. Please refer to this
    [issue](https://github.com/tensorflow/tfx/issues/838).

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
    *   specify a ComponentSpec contract and conform to new class definition
        style (see `base_component.BaseComponent`)
    *   specify `EXECUTOR_SPEC=ExecutorClassSpec(MyExecutor)` in the component
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
