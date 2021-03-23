# Current Version (Still in Development)

## Major Features and Improvements

*  Added a simple query based driver that supports Span spec and static_range.
*  Added e2e rolling window example/test for Span Resolver.
*  Performance improvement in Transform by avoiding excess encodings and
   decodings when it materializes transformed examples or generates statistics
   (both enabled by default).
*  Added an accessor (`.data_view_decode_fn`) to the decoder function wrapped in
   the DataView in Trainer `FnArgs.data_accessor`.

## Breaking Changes

### For Pipeline Authors

*  CLI requires Apache Airflow 1.10.14 or later. If you are using an older
   version of airflow, you can still copy runner definition to the DAG
   directory manually and run using airflow UIs.

### For Component Authors

## Deprecations

*   Deprecated input/output compatibility aliases for Transform and
    StatisticsGen.

## Bug Fixes and Other Changes

*   The `tfx_version` custom property of output artifacts is now set by the
    default publisher to the TFX SDK version.
*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `packaging>=20,<21`.

## Documentation Updates
*   Simplified Apache Spark and Flink example deployment scripts by using Beam's
    SparkRunner and FlinkRunner classes.
*   Upgraded example Apache Flink deployment to Flink 1.12.1.
*   Upgraded example Apache Spark deployment to Spark 2.4.7.
*   Added the "TFX Python function component" notebook tutorial.

# Version 0.28.0

## Major Features and Improvements

*   Publically released TFX docker image in [tensorflow/tfx](
    https://hub.docker.com/r/tensorflow/tfx) will use GPU
    compatible based TensorFlow images from [Deep Learning Containers](
    https://cloud.google.com/ai-platform/deep-learning-containers). This allow
    these images to be used with GPU out of box.
*   Added an example pipeline for a ranking model (using
    [tensorflow_ranking](https://github.com/tensorflow/ranking))
    at `tfx/examples/ranking`. More documentation will be available in future
    releases.
*   Added a [spans_resolver](
    https://github.com/tensorflow/tfx/blob/master/tfx/dsl/experimental/spans_resolver.py)
    that can resolve spans based on range_config.

## Breaking Changes

### For Pipeline Authors

*   Custom arg key in `google_cloud_ai_platform.tuner.executor` is renamed to
    `ai_platform_tuning_args` from `ai_platform_training_args`, to better
    distinguish usage with Trainer.

### For component authors

*   N/A

## Deprecations

*   Deprecated input/output compatibility aliases for Transform and SchemaGen.

## Bug Fixes and Other Changes

*   Change Bigquery ML Pusher to publish the model to the user specified project
    instead of the default project from run time context.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `ml-metadata>=0.28.0,<0.29.0`.
*   Depends on `kfp-pipeline-spec>=0.1.6,<0.2`.
*   Depends on `struct2tensor>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-data-validation>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-model-analysis>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-transform>=0.28.0,<0.29.0`.
*   Depends on `tfx-bsl>=0.28.1,<0.29.0`.

## Documentation Updates

*   Published a [migration instruction](
    https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/launcher/README.md)
    for legacy custom launcher developers.

# Version 0.27.0

## Major Features and Improvements

*   Updated the `tfx.components.evaluator.Evaluator` component to support
    [TFMA's "model-agnostic" evaluation](https://www.tensorflow.org/tfx/model_analysis/faq#how_do_i_setup_tfma_to_work_with_pre-calculated_ie_model-agnostic_predictions_tfrecord_and_tfexample).
    The `model` channel is now optional when constructing the component, which
    is useful when the `examples` channel provides tf.Examples containing both
    the labels and pre-computed model predictions, i.e. "model-agnostic"
    evaluation.
*   Supports different types of quantizations on TFLite conversion using
    TFLITE_REWRITER by setting `quantization_optimizations`,
    `quantization_supported_types` and `quantization_enable_full_integer`. Flag
    definitions can be found here: [Post-traning
    quantization](https://www.tensorflow.org/lite/performance/post_training_quantization).
*   Added automatic population of `tfdv.StatsOptions.vocab_paths` when computing
    statistics within the Transform component.

## Breaking changes

### For pipeline authors

*   `enable_quantization` from TFLITE_REWRITER is removed and setting
    quantization_optimizations = [tf.lite.Optimize.DEFAULT] will perform the
    same type of quantization, dynamic range quantization. Users of the
    TFLITE_REWRITER who do not enable quantization should be uneffected.
*   Default value for `infer_feature_shape` for SchemaGen changed from `False`
    to `True`, as indicated in previous release log. The inferred schema might
    change if you do not specify `infer_feature_shape`. It might leads to
    changes of the type of input features in Transform and Trainer code.

### For component authors

*   N/A

## Deprecations

*   Pipeline information is not be stored on the local filesystem anymore using
    Kubeflow Pipelines orchestration with CLI. Instead, CLI will always use the
    latest version of the pipeline in the Kubeflow Pipeline cluster. All
    operations will be executed based on the information on the Kubeflow
    Pipeline cluster. There might be some left files on
    `${HOME}/tfx/kubeflow` or `${HOME}/kubeflow` but those will not be used
    any more.
*   The `tfx.components.common_nodes.importer_node.ImporterNode` class has been
    moved to `tfx.dsl.components.common.importer.Importer`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   The `tfx.components.common_nodes.resolver_node.ResolverNode` class has been
    moved to `tfx.dsl.components.common.resolver.Resolver`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   The `tfx.dsl.resolvers.BaseResolver` class has been
    moved to `tfx.dsl.components.common.resolver.ResolverStrategy`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   Deprecated input/output compatibility aliases for ExampleValidator,
    Evaluator, Trainer and Pusher.

## Bug fixes and other changes

*   Add error condition checks to BulkInferrer's `output_example_spec`.
    Previously, when the `output_example_spec` did not include the correct spec
    definitions, the BulkInferrer would fail silently and output examples
    without predictions.
*   InfraValidator supports using alternative TensorFlow Serving image in case
    deployed environment cannot reach the public internet (nor the docker hub).
    Such alternative image should behave the same as official
    `tensorflow/serving` image such as the same model volume path, serving port,
    etc.
*   Executor in `tfx.extensions.google_cloud_ai_platform.pusher.executor`
    supported regional endpoint and machine_type.
*   Starting from this version, proto files which are used to generate
    component-level configs are included in the `tfx` package directly.
*   The `tfx.dsl.io.fileio.NotFoundError` exception unifies handling of not-
    found errors across different filesystem plugin backends.
*   Fixes the serialization of zero-valued default when using `RuntimeParameter`
    on Kubeflow.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `ml-metadata>=0.27.0,<0.28.0`.
*   Depends on `numpy>=1.16,<1.20`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `kfp-pipeline-spec>=0.1.5,<0.2` in test and image.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-data-validation>=0.27.0,<0.28.0`.
*   Depends on `tensorflow-model-analysis>=0.27.0,<0.28.0`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-transform>=0.27.0,<0.28.0`.
*   Depends on `tfx-bsl>=0.27.0,<0.28.0`.

## Documentation updates

*   N/A

# Version 0.26.1

*   This a bug fix only version

## Major Features and Improvements

*   N/A

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Deprecations

*   N/A

## Bug fixes and other changes

*   The `tfx.version` attribute was restored.

## Documentation updates

*   N/A

# Version 0.26.0

## Major Features and Improvements

*   Supported output examples artifact for BulkInferrer which can be used to
    link with downstream training.
*   TFX Transform switched to a (notably) faster and more accurate
    implementation of `tft.quantiles` analyzer.
*   Added native TF 2 implementation of Transform. The default
    behavior will continue to use Tensorflow's compat.v1 APIs. This can be
    overriden by passing `force_tf_compat_v1=False` and enabling TF 2 behaviors.
    The default behavior for TF 2 will be switched to the new native
    implementation in a future release.
*   Added support for passing a callable to set pre/post transform statistic
    generation options.
*   In addition to the "tfx" pip package, a dependency-light distribution of the
    core pipeline authoring functionality of TFX is now available as the
    "ml-pipelines-sdk" pip package. This package does not include first-party
    TFX components. The "tfx" pip package is still the recommended installation
    path for TFX.
*   Migrated LocalDagRunner to the new [IR](https://github.com/tensorflow/tfx/blob/master/tfx/proto/orchestration/pipeline.proto) stack.

## Breaking changes

*   Wheel package building for TFX has changed, and users need to follow the
    [new TFX package build instructions]
    (https://github.com/tensorflow/tfx/blob/master/package_build/README.md) to
    build wheels for TFX.


### For pipeline authors

*   Added BigQueryToElwcExampleGen to take a query as input and generate
    ExampleListWithContext (ELWC) examples.

### For component authors

*   N/A

## Deprecations

*   TrainerFnArgs is deprecated by FnArgs.
*   Deprecated DockerComponentConfig class: user should set a DockerPlatformConfig
    proto in `platform_config` using `with_platform_config()` API instead.

## Bug fixes and other changes

*   Official TFX container image's entrypoint is changed so the image can be
    used as a custom worker for Dataflow.
*   In the published TFX container image, wheel files are now used to install
    TFX, and the TFX source code has been moved to `/tfx/src`.
*   Added a skeleton of CLI support for Kubeflow V2 runner, and implemented
    support for pipeline operations.
*   Added an experimental template to use with Kubeflow V2 runner.
*   Added sanitization of user-specified pipeline name in Kubeflow V2 runner.
*   Migrated `deployment_config` in Kubeflow V2 runner from `Any` proto message
    to `Struct`, to ensure compatibility across different copies of the proto
    libraries.
*   The `tfx.dsl.io.fileio` filesystem handler will delegate to
    `tensorflow.io.gfile` for any unknown filesystem schemes if TensorFlow
    is installed.
*   Skipped ephemeral package when the beam flag
    'worker_harness_container_image' is set.
*   The `tfx.dsl.io.makedirs` call now succeeds if the directory already exists.
*   Fixed the component entrypoint, so that it creates the parent directory for
    the output metadata file before trying to write the data.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26,<3`.
*   Depends on `keras-tuner>=1,<1.0.2`.
*   Depends on `kfp-pipeline-spec>=0.1.3,<0.2`.
*   Depends on `ml-metadata>=0.26.0,<0.27.0`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-data-validation>=0.26,<0.27`.
*   Depends on `tensorflow-model-analysis>=0.26,<0.27`.
*   Depends on `tensorflow-serving>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-transform>=0.26,<0.27`.
*   Depends on `tfx-bsl>=0.26.1,<0.27`.

## Documentation updates

*   N/A

# Version 0.25.0

## Major Features and Improvements

*   Supported multiple artifacts for Transform's input example and output
    transformed example channels.
*   Added support for processing specific spans in file-based ExampleGen with
    range configuration.
*   Added ContainerExecutableSpec in portable IR to support container components
    portable orchestrator.
*   Added Placeholder utility library. Placeholder can be used to represent
    not-yet-available value at pipeline authoring time.
*   Added support for the `tfx.dsl.io.fileio` pluggable filesystem interface,
    with initial support for local files and the Tensorflow GFile filesystem
    implementation.
*   SDK and example code now uses `tfx.dsl.io.fileio` instead of `tf.io.gfile`
    when possible for filesystem I/O implementation portability.
*   From this release TFX will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install -i https://pypi-nightly.tensorflow.org/simple tfx
    ```
    Note: These nightly packages are unstable and breakages are likely to happen.
    The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFX available on PyPI by running the
    command
    ```
    pip install tfx
    ```
*   Added CloudTuner KFP e2e example running on Google Cloud Platform with
    distributed tuning.
*   Migrated BigQueryExampleGen to the new `ReadFromBigQuery` on all runners.
*   Introduced Kubeflow V2 DAG runner, which is based on
    [Kubeflow IR spec](https://github.com/kubeflow/pipelines/blob/master/api/v2alpha1/pipeline_spec.proto).
    Same as `KubeflowDagRunner` it will compile the DSL pipeline into a payload
    but not trigger the execution locally.
*   Added compile time check for schema mismatch in Kubeflow V2 runner.
*   Added 'penguin' example. Penguin example uses Palmer Penguins dataset and
    classify penguin species using four numeric features.
*   Iris e2e examples are replaced by penguin examples.
*   TFX BeamDagRunner is migrated to use the tech stack built on top of [IR](https://github.com/tensorflow/tfx/blob/master/tfx/proto/orchestration/pipeline.proto).
    While this is no-op to users, it is a major step towards supporting more
    flexible TFX DSL [semetic](https://github.com/tensorflow/community/blob/master/rfcs/20200601-tfx-udsl-semantics.md).
    Please refer to the [RFC](https://github.com/tensorflow/community/blob/master/rfcs/20200705-tfx-ir.md)
    of IR to learn more details.
*   Supports forward compatibility when evolving TFX artifact types, which
    allows jobs of old release and new release run with the same MLMD instance.
*   Graduated the portable/beam_dag_runner.py to beam/beam_dag_runner.py


## Breaking changes

*   Moved the directory that CLI stores pipeline information from
    ${HOME}/${ORCHESTRATOR} to ${HOME}/tfx/${ORCHESTRATOR}. For example,
    "~/kubeflow" was changed to "~/tfx/kubeflow". This directory is used to
    store pipeline information including pipeline ids in the Kubeflow Pipelines
    cluster which are needed to create runs or update pipelines.
    These files will be moved automatically when it is first used and no
    separate action is needed.
    See https://github.com/tensorflow/tfx/blob/master/docs/guide/cli.md for the
    detail.

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Deprecations

*   Modules under `tfx.components.base` have been deprecated and moved to
    `tfx.dsl.components.base` in preparation for releasing a pipeline authoring
    package without explicit Tensorflow dependency.
*   Deprecated setting `instance_name` at pipeline node level. Instead, users
    are encouraged to set `id` explicitly of any pipeline node through newly
    added APIs.

## Bug fixes and other changes

*   Added the LocalDagRunner to allow local pipeline execution without using
    Apache Beam. This functionality is in development.
*   Introduced dependency to `tensorflow-cloud` Python package, with intention
    to separate out Google Cloud Platform specific extensions.
*   Depends on `mmh>=2.2,<3` in container image for potential performance
    improvement for Beam based hashes.
*   New extra dependencies `[examples]` is required to use codes inside
    tfx/examples.
*   Fixed the run_component script.
*   Stopped depending on `WTForms`.
*   Fixed an issue with Transform cache and beam 2.24-2.25 in an interactive
    notebook that caused it to fail.
*   Scripts - run_component - Added a way to output artifact properties.
*   Fixed an issue resulting in incorrect cache miss to ExampleGen when no
    `beam_pipeline_args` is provided.
*   Changed schema as an optional input channel of Trainer as schema can be
    accessed from TFT graph too.
*   Fixed an issue during recording of a component's execution where
    "missing or modified key in exec_properties" was raised from MLMD when
    `exec_properties` both omitted an existing property and added a new
    property.
*   Supported users to set `id` of pipeline nodes directly.
*   Added a new template, 'penguin' which is simple subset of
    [penguin examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples/penguin),
    and uses the same
    [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)
    dataset. The new template focused on easy ingestion of user's own data.
*   Changed default data path for the taxi template from `tfx-template/data`
    to `tfx-template/data/taxi`.
*   Fixed a bug which crashes the pusher when infra validation did not pass.
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `attrs>=19.3.0,<21`.
*   Depends on `kfp-pipeline-spec>=0.1.2,<0.2`.
*   Depends on `kfp>=1.1.0,<2`.
*   Depends on `ml-metadata>=0.25,<0.26`.
*   Depends on `tensorflow-cloud>=0.1,<0.2`.
*   Depends on `tensorflow-data-validation>=0.25,<0.26`.
*   Depends on `tensorflow-hub>=0.9.0,<0.10`.
*   Depends on `tensorflow-model-analysis>=0.25,<0.26`.
*   Depends on `tensorflow-transform>=0.25,<0.26`.
*   Depends on `tfx-bsl>=0.25,<0.26`.

## Documentation updates

*   N/A

# Version 0.24.1

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Fixes issues where custom property access of a missing property created an invalid MLMD Artifact protobuf message.

### Deprecations

*   N/A

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

# Version 0.24.0

## Major Features and Improvements

*   Use TFXIO and batched extractors by default in Evaluator.
*   Supported custom split configuration for ExampleGen and its downstream
    components. Instead of hardcoded 'train' and 'eval' splits, TFX components
    now can process the custom splits generated by ExampleGen. For details,
    please refer to [ExampleGen doc](https://github.com/tensorflow/tfx/blob/r0.24.0/docs/guide/examplegen.md#custom-examplegen)
*   Added python 3.8 support.

## Bug fixes and other changes

*   Supported CAIP Runtime 2.2 for online prediction pusher.
*   Used 'python -m ' style for container entrypoints.
*   Stopped depending on `google-resumable-media`.
*   Stopped depending on `Werkzeug`.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `apache-beam[gcp]>=2.24,<3`.
*   Depends on `ml-metadata>=0.24,<0.25`.
*   Depends on `protobuf>=3.12.2,<4`.
*   Depends on `tensorflow-data-validation>=0.24.1,<0.25`.
*   Depends on `tensorflow-model-analysis>=0.24.3,<0.25`.
*   Depends on `tensorflow-transform>=0.24.1,<0.25`.
*   Depends on `tfx-bsl>=0.24.1,<0.25`.

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

## Deprecations

*   Deprecated python 3.5 support.

# Version 0.23.1
*   This is a bug fix version (to resolve impossible dependency conflicts).
## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Stopped depending on `google-resumable-media`.
*   Depends on `apache-beam[gcp]>=2.24,<3`.
*   Depends on `tensorflow-data-validation>=0.23.1,<0.24`.

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

## Deprecations

*   Deprecated Python 3.5 support.

# Version 0.23.0

## Major Features and Improvements
*   Added TFX DSL IR compiler that encodes a TFX pipeline into a DSL proto.
*   Supported feature based split partition in ExampleGen.
*   Added the ConcatPlaceholder to tfx.dsl.component.experimental.placeholders.
*   Changed Span information as a property of ExampleGen's output artifact.
    Deprecated ExampleGen input (external) artifact.
*   Added ModelRun artifact for Trainer for storing training related files,
    e.g., Tensorboard logs. Trainer's Model artifact now only contain pure
    models (check `tfx/utils/path_utils.py` for details).
*   Added support for `tf.train.SequenceExample` in ExampleGen:
    *   ImportExampleGen now supports `tf.train.SequenceExample` importing.
    *   base_example_gen_executor now supports `tf.train.SequenceExample` as
        output payload format, which can be utilized by custom ExampleGen.
*   Added Tuner component and its integration with Google Cloud Platform as
    the execution and hyperparemeter optimization backend.
*   Switched Transform component to use the new TFXIO code path. Users may
    potentially notice large performance improvement.
*   Added support for primitive artifacts to InputValuePlaceholder.
*   Supported multiple artifacts for Trainer and Tuner's input example Channel.
*   Supported split configuration for Trainer and Tuner.
*   Supported split configuration for Evaluator.
*   Supported split configuration for StatisticsGen, SchemaGen and
    ExampleValidator. SchemaGen will now use all splits to generate schema
    instead of just using `train` split. ExampleValidator will now validate all
    splits against given schema instead of just validating `eval` split.
*   Component authors now can create a TFXIO instance to get access to the
    data through `tfx.components.util.tfxio_utils`. As TFX is going to
    support more data payload formats and data container formats, using
    `tfxio_utils` is encouraged to avoid dealing directly with each combination.
    TFXIO is the interface of [Standardized TFX Inputs](
    https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md).
*   Added experimental BaseStubExecutor and StubComponentLauncher to test TFX
    pipelines.
*   Added experimental TFX Pipeline Recorder to record output artifacts of the
    pipeline.
*   Supported multiple artifacts in an output Channel to match a certain input
    Channel's artifact count. This enables Transform component to process
    multiple artifacts.
*   Transform component's transformed examples output is now optional (enabled
    by default). This can be disabled by specifying parameter
    `materialize=False` when constructing the component.
*   Supported `Version` spec in input config for file based ExampleGen.
*   Added custom config to Transform component and made it available to
    pre-processing fn.
*   Supported custom extractors in Evaluator.
*   Deprecated tensorflow dependency from MLMD python client.
*   Supported `Date` spec in input config for file based ExampleGen.
*   Enabled analyzer cache optimization in the Transform component:
    *   specify `analyzer_cache` to use the cache generated from a previous run.
    *   specify parameter `disable_analyzer_cache=True` (False by default) to
        disable cache (won't generate cache output).
*   Added support for width modifiers in Span and Version specs for file based
    ExampleGen.

## Bug fixes and other changes
*   Added Tuner component to Iris e2e example.
*   Relaxed the rule that output artifact uris must be newly created. This is a
    temporary workaround to make retry work. We will introduce a more
    comprehensive solution for idempotent execution.
*   Made evaluator output optional (while still recommended) for pusher.
*   Moved BigQueryExampleGen to `tfx.extensions.google_cloud_big_query`.
*   Moved BigQuery ML Pusher to `tfx.extensions.google_cloud_big_query.pusher`.
*   Removed Tuner from custom_components/ as it's supported under components/
    now.
*   Added support of non tf.train.Example protos as internal data payload
    format by ImportExampleGen.
*   Used thread local storage for `label_utils.scoped_labels()` to make it
    thread safe.
*   Requires [Bazel](https://bazel.build/) to build TFX source code.
*   Upgraded python version in TFX docker images to 3.7. Older version of
    python (2.7/3.5/3.6) is not available anymore in `tensorflow/tfx` images
    on docker hub. Virtualenv is not used anymore.
*   Stopped requiring `avro-python3`.
*   Depends on `absl-py>=0.7,<0.9`.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `attrs>=19.3.0,<20`.
*   Depends on `ml-metadata>=0.23,<0.24`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
    * Note: Dependency like `tensorflow-transform` might impose a narrower
      range of `tensorflow`.
*   Depends on `tensorflow-data-validation>=0.23,<0.24`.
*   Depends on `tensorflow-model-analysis>=0.23,<0.24`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-transform>=0.23,<0.24`.
*   Depends on `tfx-bsl>=0.23,<0.24`.
*   Added execution_result_pb2.ExecutorOutput as an Optional return value of
    BaseExecutor. This change is backward compatible to all existing executors.
*   Added executor_output_uri and stateful_working_dir to Executor's context.

## Breaking changes
*   Changed the URIs of the value artifacts to point to files.
*   De-duplicated the
    tfx.dsl.component.experimental.executor_specs.CommandLineArgumentType
    union type in favor of
    tfx.dsl.component.experimental.placeholders.CommandLineArgumentType


### For pipeline authors
*   Moved BigQueryExampleGen to `tfx.extensions.google_cloud_big_query`. The
    previous module path from `tfx.components` is not available anymore. This is
    a breaking change.
*   Moved BigQuery ML Pusher to `tfx.extensions.google_cloud_big_query.pusher`.
    The previous module path from `tfx.extensions.google_cloud_big_query_ml`
    is not available anymore.
*   Updated beam pipeline args, users now need to set both `direct_running_mode`
    and `direct_num_workers` explicitly for multi-processing.
*   Added required 'output_data_format' execution property to
    FileBaseExampleGen.
*   Changed ExampleGen to take a string as input source directly instead of a
    Channel of external artifact:
    *   Previously deprecated `input_base` Channel is changed to string type
        instead of Channel. This is a breaking change, users should pass string
        directly to `input_base`.
*   Fully removed csv_input and tfrecord_input in dsl_utils. This is a breaking
    change, users should pass string directly to `input_base`.

### For component authors
*   Changed GetInputSourceToExamplePTransform interface by removing input_dict.
    This is a breaking change, custom ExampleGens need to follow the interface
    change.
*   Changed ExampleGen to take a string as input source directly instead of a
    Channel of external artifact:
    *   `input` Channel is deprecated. The use of `input` is valid but
        should change to string type `input_base` ASAP.

## Documentation updates
* N/A

## Deprecations
*   ExternalArtifact and `external_input` function are deprecated. The use
    of `external_input` with ExampleGen `input` is still valid but should change
    to use `input_base` ASAP.
*   Note: We plan to remove Python 3.5 support after this release.

# Version 0.22.2

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Reuse Examples artifact type introduced in TFX 0.23 to allow older release jobs running together with TFX 0.23+ release.

### Deprecations

*   N/A

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

# Version 0.22.1

## Major Features and Improvements
* N/A

## Bug fixes and other changes
*   Depends on 'tensorflowjs>=2.0.1.post1,<3' for `[all]` dependency.
*   Fixed the name of the usage telemetry when tfx templates are used.
*   Depends on `tensorflow-data-validation>=0.22.2,<0.23.0`.
*   Depends on `tensorflow-model-analysis>=0.22.2,<0.23.0`.
*   Depends on `tfx-bsl>=0.22.1,<0.23.0`.
*   Depends on `ml-metadata>=0.22.1,<0.23.0`.

## Breaking changes

### For pipeline authors
* N/A

### For component authors
* N/A

## Documentation updates
* N/A

## Deprecations
* N/A

# Version 0.22.0

## Major Features and Improvements
*   Introduced experimental Python function component decorator (`@component`
    decorator under `tfx.dsl.component.experimental.decorators`) allowing
    Python function-based component definition.
*   Added the experimental TemplatedExecutorContainerSpec executor class that
    supports structural placeholders (not Jinja placeholders).
*   Added the experimental function "create_container_component" that
    simplifies creating container-based components.
*   Implemented a TFJS rewriter.
*   Added the scripts/run_component.py script which makes it easy to run the
    component code and executor code. (Similar to scripts/run_executor.py)
*   Added support for container component execution to BeamDagRunner.
*   Introduced experimental generic Artifact types for ML workflows.
*   Added support for `float` execution properties.

## Bug fixes and other changes
*   Migrated BigQueryExampleGen to the new (experimental) `ReadFromBigQuery`
    PTramsform when not using Dataflow runner.
*   Enhanced add_downstream_node / add_upstream_node to apply symmetric changes
    when being called. This method enables task-based dependencies by enforcing
    execution order for synchronous pipelines on supported platforms. Currently,
    the supported platforms are Airflow, Beam, and Kubeflow Pipelines. Note that
    this API call should be considered experimental, and may not work with
    asynchronous pipelines, sub-pipelines and pipelines with conditional nodes.
*   Added the container-based sample pipeline (download, filter, print)
*   Removed the incomplete cifar10 example.
*   Removed `python-snappy` from `[all]` extra dependency list.
*   Tests depends on `apache-airflow>=1.10.10,<2`;
*   Removed test dependency to tzlocal.
*   Fixes unintentional overriding of user-specified setup.py file for Dataflow
    jobs when running on KFP container.
*   Made ComponentSpec().inputs and .outputs behave more like real dictionaries.
*   Depends on `kerastuner>=1,<2`.
*   Depends on `pyyaml>=3.12,<6`.
*   Depends on `apache-beam[gcp]>=2.21,<3`.
*   Depends on `grpcio>=2.18.1,<3`.
*   Depends on `kubernetes>=10.0.1,<12`.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`.
*   Depends on `tensorflow-data-validation>=0.22.0,<0.23.0`.
*   Depends on `tensorflow-model-analysis>=0.22.1,<0.23.0`.
*   Depends on `tensorflow-transform>=0.22.0,<0.23.0`.
*   Depends on `tfx-bsl>=0.22.0,<0.23.0`.
*   Depends on `ml-metadata>=0.22.0,<0.23.0`.
*   Depends on 'tensorflowjs>=2.0.1.post1,<3' for `[all]` dependency.
*   Fixed a bug in `io_utils.copy_dir` which prevent it to work correctly for
    nested sub-directories.

## Breaking changes

### For pipeline authors
*   Changed custom config for the Do function of Trainer and Pusher to accept
    a JSON-serialized dict instead of a dict object. This also impacts all the
    Do functions under `tfx.extensions.google_cloud_ai_platform` and
    `tfx.extensions.google_cloud_big_query_ml`. Note that this breaking
    change occurs at the signature of the executor's Do function. Therefore, if
    the user did not customize the Do function, and the compile time SDK version
    is aligned with the run time SDK version, previous pipelines should still
    work as intended. If the user is using a custom component with customized
    Do function, `custom_config` should be assumed to be a JSON-serialized
    string from next release.
*   For users of BigQueryExampleGen, `--temp_location` is now a required Beam
    argument, even for DirectRunner. Previously this argument was only required
    for DataflowRunner. Note that the specified value of `--temp_location`
    should point to a Google Cloud Storage bucket.
*   Revert current per-component cache API (with `enable_cache`, which was only
    available in tfx>=0.21.3,<0.22), in preparing for a future redesign.

### For component authors
*   Converted the BaseNode class attributes to the constructor parameters. This
    won't affect any components derived from BaseComponent.
*   Changed the encoding of the Integer and Float artifacts to be more portable.

## Documentation updates
*   Added concept guides for understanding TFX pipelines and components.
*   Added guides to building Python function-based components and
    container-based components.
*   Added BulkInferrer component and TFX CLI documentation to the table of
    contents.

## Deprecations
*   Deprecating Py2 support

# Version 0.21.5

## Major Features and Improvements

*   N/A

## Bug fixes and other changes

*   Reuse Examples artifact type introduced in TFX 0.23 to allow older release jobs running together with TFX 0.23+ release.
*   Removed python-snappy from [all] extra dependency list.

### Deprecations

*   N/A

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

# Version 0.21.4

## Major Features and Improvements

## Bug fixes and other changes
*   Fixed InfraValidator signal handling bug on BeamDagRunner.
*   Dropped "Type" suffix from primitive type artifact names (Integer, Float,
    String, Bytes).

### Deprecations

## Breaking changes

### For pipeline authors

### For component authors

## Documentation updates

# Version 0.21.3

## Major Features and Improvements
*   Added run/pipeline link when creating runs/pipelines on KFP through TFX CLI.
*   Added support for `ValueArtifact`, whose attribute `value` allows users to
    access the content of the underlying file directly in the executor. Support
    Bytes/Integer/String/Float type. Note: interactive resolution does not
    support this for now.
*   Added InfraValidator component that is used as an early warning layer
    before pushing a model into production.

## Bug fixes and other changes
*   Starting this version, TFX will only release python3 packages.
*   Replaced relative import with absolute import in generated templates.
*   Added a native keras model in the taxi template and the template now uses
    generic Trainer.
*   Added support of TF 2.1 runtime configuration for AI Platform Prediction
    Pusher.
*   Added support for using ML Metadata ArtifactType messages as Artifact
    classes.
*   Changed CLI behavior to create new versions of pipelines instead of
    delete and create new ones when pipelines are updated for KFP. (Requires
    kfp >= 0.3.0)
*   Added ability to enable quantization in tflite rewriter.
*   Added k8s pod labels when the pipeline is executed via KubeflowDagRunner for
    better usage telemetry.
*   Parameterized the GCP taxi pipeline sample for easily ramping up to full
    taxi dataset.
*   Added support for hyphens(dash) in addition to underscores in CLI flags.
    Underscores will be supported as well.
*   Fixed ill-formed underscore in the markdown visualization when running on
    KFP.
*   Enabled per-component control for caching with enable_cache argument in
    each component.

### Deprecations

## Breaking changes

### For pipeline authors

### For component authors

## Documentation updates

# Version 0.21.2

## Major Features and Improvements
*   Updated `StatisticsGen` to optionally consume a schema `Artifact`.
*   Added support for configuring the `StatisticsGen` component via serializable
    parts of `StatsOptions`.
*   Added Keras guide doc.
*   Changed Iris model_to_estimator e2e example to use generic Trainer.
*   Demonstrated how TFLite is supported in TFX by extending MNIST example
    pipeline to also train a TFLite model.

## Bug fixes and other changes
*   Fix the behavior of Trainer Tensorboard visualization when caching is used.
*   Added component documentation and guide on using TFLite in TFX.
*   Relaxed the PyYaml dependency.

### Deprecations
*   Model Validator (its functionality is now provided by the Evaluator).

## Breaking changes

### For pipeline authors

### For component authors

## Documentation updates

# Version 0.21.1

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
*   Added a Chicago Taxi example with native Keras.
*   Updated TFLite converter to work with TF2.
*   Enabled filtering by artifact producer and output key in ResolverNode.

## Bug fixes and other changes
*   Added --skaffold_cmd flag when updating a pipeline for kubeflow in CLI.
*   Changed python_version to 3.7 when using TF 1.15 and later for Cloud AI Platform Prediction.
*   Added 'tfx_runner' label for CAIP, BQML and Dataflow jobs submitted from
    TFX components.
*   Fixed the Taxi Colab notebook.
*   Adopted the generic trainer executor when using CAIP Training.
*   Depends on 'tensorflow-data-validation>=0.21.4,<0.22'.
*   Depends on 'tensorflow-model-analysis>=0.21.4,<0.22'.
*   Depends on 'tensorflow-transform>=0.21.2,<0.22'.
*   Fixed misleading logs in Taxi pipeline portable Beam example.

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
*   Added experimental cli option `template`, which can be used to scaffold a
    new pipeline from TFX templates. Currently the `taxi` template is provided
    and more templates would be added in future versions.
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
*   Added an iris example with native Keras.
*   Added an MNIST example with native Keras.

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
*   Use '_' instead of '/' in feature names of several examples to avoid
    potential clash with namescope separator.


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
