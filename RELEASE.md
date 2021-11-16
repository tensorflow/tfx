# Version 1.4.0

## Major Features and Improvements

*   Supported endpoint overwrite for CAIP BulkInferrer.
*   Added support for outputting and encoding `tf.RaggedTensor`s in TFX
    Transform component.
*   Added conditional for TFX running on KFPv2 (Vertex).
*   Supported component level beam pipeline args for Vertex (KFPV2DagRunner).
*   Support exit handler for TFX running on KFPv2 (Vertex).
*   Added RangeConfig for QueryBasedExampleGen to select date using query
    pattern.
*   Added support for union of Channels as input to standard TFX components.
    Users can use channel.union() to combine multiple Channels and use as input
    to these compnents. Artfacts resolved from these channels are expected to
    have the same type, and passed to components in no particular order.

## Breaking Changes

*   Calling `TfxRunner.run(pipeline)` with the Pipeline IR proto will no longer
    be supported. Please switch to `TfxRunner.run_with_ir(pipeline)` instead.
    If you are calling `TfxRunner.run(pipeline)` with the Pipeline object, this
    change should not affect you.

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   Deprecated python3.6 support.

## Bug Fixes and Other Changes

*   Depends on `google-cloud-aiplatform>=1.5.0,<2`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<2.7`.
*   Depends on `pyarrow>=1,<6`.
*   Fixed FileBasedExampleGen driver for Kubeflow v2 (Vertex). Driver can
    update exec_properties for its executor now, which enables {SPAN} feature.
*   example_gen.utils.dict_to_example now accepts Numpy types
*   Updated pytest to include v6.x
*   Depends on `apache-beam[gcp]>=2.33,<3`.
*   Depends on `ml-metadata>=1.4.0,<1.5.0`.
*   Depends on `struct2tensor>=0.35.0,<0.36.0`.
*   Depends on `tensorflow-data-validation>=1.4.0,<1.5.0`.
*   Depends on `tensorflow-model-analysis>=0.35.0,<0.36.0`.
*   Depends on `tensorflow-transform>=1.4.0,<1.5.0`.
*   Depends on `tfx-bsl>=1.4.0,<1.5.0`.
*   Fixed error where Vertex Endpoints of the same name is not deduped

## Documentation Updates

*   N/A

