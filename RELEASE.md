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
*   Fixed a bug in `io_utils.copy_dir` which prevent it to work correctly for
    nested sub-directories.
*   Fixed the name of the usage telemetry when tfx templates are used.

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
