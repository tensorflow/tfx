# Version 1.2.0

## Major Features and Improvements

*  Added RuntimeParam support for Trainer's custom_config.
*  TFX Trainer and Pusher now support Vertex, which can be enabled with
   `ENABLE_VERTEX_KEY` key in `custom_config`.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Fixed ths issue that kfp_pod_name is not generated as an execution property
    for Kubeflow Pipelines.
*   Fixed issue when InputValuePlaceholder is used as component parameter in
    container based component.
*   Depends on `kubernetes>=10.0.1,<13`
*   `CsvToExample` now supports multi-line strings.
*   `tfx.benchmarks` package was removed from the Python TFX wheel. This package
    is used only for benchmarking and not useful for end users.
*   Fixed the issue for fairness_indicator_thresholds support of Evaluator.
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `kfp-pipeline-spec>=0.1.8,<0.2`.
*   Depends on `ml-metadata>=1.2.0,<1.3.0`.
*   Depends on `struct2tensor>=0.33.0,<0.34.0`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<2.6`.
*   Depends on `tensorflow-data-validation>=1.2.0,<1.3.0`.
*   Depends on `tensorflow-model-analysis>=0.33.0,<0.34.0`.
*   Depends on `tensorflow-transform>=1.2.0,<1.3.0`.
*   Depends on `tfx-bsl>=1.2.0,<1.3.0`.

## Documentation Updates

*   N/A
