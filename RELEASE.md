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
*   Supported split configuration for Transform.
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

## Breaking changes
*   Changed the URIs of the value artifacts to point to files.

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

