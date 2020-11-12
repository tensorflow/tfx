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
*   Added the LocalDagRunner to allow local pipeline execution without using
    Apache Beam.
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
*   Supports forward compatibility when evolving TFX artifact types, which
    allows jobs of old release and new release run with the same MLMD instance.

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Deprecations

*   Modules under `tfx.components.base` have been deprecated and moved to
    `tfx.dsl.components.base` in preparation for releasing a pipeline authoring
    package without explicit Tensorflow dependency.

## Bug fixes and other changes

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
*   Depends on `apache-beam[gcp]>=2.25,<3`.
*   Depends on `attrs>=19.3.0,<21`.
*   Depends on `ml-metadata>=0.25,<0.26`.
*   Depends on `tensorflow-cloud>=0.1,<0.2`.
*   Depends on `tensorflow-data-validation>=0.25,<0.26`.
*   Depends on `tensorflow-hub>=0.9.0,<0.10`.
*   Depends on `tensorflow-model-analysis>=0.25,<0.26`.
*   Depends on `tensorflow-transform>=0.25,<0.26`.
*   Depends on `tfx-bsl>=0.25,<0.26`.

## Documentation updates

*   N/A
