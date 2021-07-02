# Version 1.0.0

## Major Features and Improvements

*  Added tfx.v1 Public APIs, please refer to
   [API doc](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1) for details.
*  Transform component now computes pre-transform and post-transform statistics
   and stores them in new, indvidual outputs ('pre_transform_schema',
   'pre_transform_stats', 'post_transform_schema', 'post_transform_stats',
   'post_transform_anomalies'). This can be disabled by setting
   `disable_statistics=True` in the Transform component.
*  BERT cola and mrpc examples now demonstrate how to calculate statistics for
   NLP features.
*  TFX CLI now supports
   [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).
   use it with `--engine=vertex` flag.
*  Telemetry: Only first-party tfx component's executor telemetry will be
   collected. All other executors will be recorded as `third_party_executor`.
   For labels longer than 63, keep first 63 characters (instead of last 63
   characters before).
*  Supports text type (use proto json string format) RuntimeParam for protos.

## Breaking Changes

*  Removed unneccessary default values for required component input Channels.
*  The `_PropertyDictWrapper` internal wrapper for `component.inputs` and
   `component.outputs` was removed: `component.inputs` and `component.outputs`
   are now unwrapped dictionaries, and the attribute accessor syntax (e.g.
   `components.outputs.output_name`) is no longer supported. Please use the
   dictionary indexing syntax (e.g. `components.outputs['output_name']`)
   instead.

### For Pipeline Authors

*   N/A

### For Component Authors

*   Apache Beam support is migrated from TFX Base Components and Executors to
    dedicated Beam Components and Executors. `BaseExecutor` will no longer embed
    `beam_pipeline_args`. Custom executors for Beam powered components should
    now extend BaseBeamExecutor instead of BaseExecutor.

## Deprecations

*   Deprecated nested RuntimeParam for Proto, Please use text type (proto json
    string) RuntimeParam instead of Proto dict with nested RuntimeParam in it.

## Bug Fixes and Other Changes

*   Forces keyword arguments for AirflowComponent to make it compatible with
    Apache Airflow 2.1.0 and later.
*   Removed `six` dependency.
*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `ml-metadata>=1.0.0,<1.1.0`.
*   Depends on `struct2tensor>=0.31.0,<0.32.0`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflow-data-validation>=1.0.0,<1.1.0`.
*   Depends on `tensorflow-hub>=0.9.0,<0.13`.
*   Depends on `tensorflowjs>=3.6.0,<4`.
*   Depends on `tensorflow-model-analysis>=0.31.0,<0.32.0`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflow-transform>=1.0.0,<1.1.0`.
*   Depends on `tfx-bsl>=1.0.0,<1.1.0`.

## Documentation Updates

*   Update the Guide of TFX to adopt 1.0 API.
*   TFT and TFDV component documentation now describes how to
    configure pre-transform and post-transform statistics, which can be used for
    validating text features.
