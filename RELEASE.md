# Version 1.0.0

## Major Features and Improvements

*  Added tfx.v1 Public APIs, please refer to
   [API doc](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1) for details.
*  Transform component now computes pre-transform and post-transform statistics
   by default. This can be disabled by setting `disable_statistics=True` in the
   Transform component.
*  BERT cola and mrpc examples now demonstrate how to calculate statistics for
   NLP features.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   Apache Beam support is migrated from TFX Base Components and Executors to
    dedicated Beam Components and Executors. `BaseExecutor` will no longer embed
    `beam_pipeline_args`. Custom executors for Beam powered components should
    now extend BaseBeamExecutor instead of BaseExecutor.

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Removed `six` dependency.
*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.5.*,<3`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.5.*,<3`.

## Documentation Updates

*   Update the Guide of TFX to adopt 1.0 API.
*   TFT and TFDV component documentation now describes how to
    configure pre-transform and post-transform statistics, which can be used for
    validating text features.
