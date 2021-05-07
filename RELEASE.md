# Version 0.30.0

## Major Features and Improvements

*  Upgraded TFX to KFP compiler to use KFP IR schema version 2.0.0.
*  InfraValidator can now produce a [SavedModel with warmup requests](
   https://www.tensorflow.org/tfx/serving/saved_model_warmup). This feature is
   enabled by setting `RequestSpec.make_warmup = True`. The SavedModel will be
   stored in the InfraBlessing artifact (`blessing` output of InfraValidator).
*  Pusher's `model` input is now optional, and `infra_blessing` can be used
   instead to push the SavedModel with warmup requests, produced by an
   InfraValidator. Note that InfraValidator does not always create a SavedModel,
   and the producer InfraValidator must be configured with
   `RequestSpec.make_warmup = True` in order to be pushed by a Pusher.
*  Support is added for the JSON_VALUE artifact property type, allowing storage
   of JSON-compatible objects as artifact metadata.
*  Support is added for the KFP v2 artifact metadata field when executing using
   the KFP v2 container entrypoint.
*  InfraValidator for Kubernetes now can override Pod manifest to customize
   annotations and environment variables.
*  Allow Beam pipeline args to be extended by specifying
   `beam_pipeline_args` per component.
*  Support string RuntimeParameters on Airflow.

## Breaking Changes

### For Pipeline Authors

*  CLI usage with kubeflow changed significantly. You MUST use the new:
  *  `--build-image` to build a container image when
     updating a pipeline with kubeflow engine.
  *  `--build-target-image` flag in CLI is changed to `--build-image` without
     any container image argument. TFX will auto detect the image specified in
     the KubeflowDagRunnerConfig class instance. For example,
     ```python
     tfx pipeline create --pipeline-path=runner.py --endpoint=xxx --build-image
     tfx pipeline update --pipeline-path=runner.py --endpoint=xxx --build-image
     ```
  *  `--package-path` and `--skaffold_cmd` flags were deleted. The compiled path
     can be specified when creating a KubeflowDagRunner class instance. TFX CLI
     doesn't depend on skaffold any more and use Docker SDK directly.
*  Default orchestration engine of CLI was changed to `local` orchestrator from
   `beam` orchestrator. You can still use `beam` orchestrator with
   `--engine=beam` flag.
*  Trainer now uses GenericExecutor as default. To use the previous Estimator
   based Trainer, please set custom_executor_spec to trainer.executor.Executor.
*  Changed the pattern spec supported for QueryBasedDriver:
   *   @span_begin_timestamp: Start of span interval, Timestamp in seconds.
   *   @span_end_timestamp: End of span interval, Timestamp in seconds.
   *   @span_yyyymmdd_utc: STRING with format, e.g., '20180114', corresponding
                           to the span interval begin in UTC.
*  Removed the already deprecated compile() method on Kubeflow V2 Dag Runner.
*  Removed project_id argument from KubeflowV2DagRunnerConfig which is not used
   and meaningless if not used with GCP.
*  Removed config from LocalDagRunner's constructor, and dropped pipeline proto
   support from LocalDagRunner's run function.
*  Removed input parameter in ExampleGen constructor and external_input in
   dsl_utils, which were called as deprecated in TFX 0.23.
*  Changed the storage type of `span` and `version` custom property in Examples
   artifact from string to int.
*  `ResolverStrategy.resolve_artifacts()` method signature has changed to take
   `ml_metadata.MetadataStore` object as the first argument.
*  Artifacts param is deprecated/ignored in Channel constructor.
*  Removed matching_channel_name from Channel's constructor.
*  Deleted all usages of instance_name, which was deprecated in version 0.25.0.
   Please use .with_id() method of components.
*  Removed output channel overwrite functionality from all official components.

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   New extra dependencies for convenience.
    - tfx[airflow] installs all Apache Airflow orchestrator dependencies.
    - tfx[kfp] installs all Kubeflow Pipelines orchestrator dependencies.
    - tfx[tf-ranking] installs packages for TensorFlow Ranking.
      NOTE: TensorFlow Ranking only compatible with TF >= 2.0.
*   Depends on 'google-cloud-bigquery>=1.28.0,<3'. (This was already installed as
    a transitive dependency from the first release of TFX.)
*   Depends on `google-cloud-aiplatform>=0.5.0,<0.8`.
*   Depends on `ml-metadata>=0.30.0,<0.31.0`.
*   Depends on `portpicker>=1.3.1,<2`.
*   Depends on `struct2tensor>=0.30.0,<0.31.0`.
*   Depends on `tensorflow-data-validation>=0.30.0,<0.31.0`.
*   Depends on `tensorflow-model-analysis>=0.30.0,<0.31.0`.
*   Depends on `tensorflow-transform>=0.30.0,<0.31.0`.
*   Depends on `tfx-bsl>=0.30.0,<0.31.0`.

## Documentation Updates

*   N/A
