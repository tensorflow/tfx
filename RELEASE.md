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
