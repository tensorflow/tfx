# Version 1.6.0

## Major Features and Improvements

*   Added experimental support for TensorFlow Decision Forests models.
*   Added Boolean type value artifacts.
*   Function components defined with `@component` may now have optional/nullable
    primitive type return values when `Optional[T]` is used in the return type
    OutputDict.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes
*   Depends on `numpy>=1.16,<2`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on `apache-beam[gcp]>=2.35,<3`.
*   Depends on `ml-metadata>=1.6.0,<1.7.0`.
*   Depends on `struct2tensor>=0.37.0,<0.38.0`.
*   Depends on `tensorflow-data-validation>=1.6.0,<1.7.0`.
*   Depends on `tensorflow-model-analysis>=0.37.0,<0.38.0`.
*   Depends on `tensorflow-transform>=1.6.0,<1.7.0`.
*   Depends on `tfx-bsl>=1.6.0,<1.7.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `kfp>=1.8.5,<2'`.
*   Pusher now copies the `saved_model.pb` file at last to prevent loading
    SavedModel on invalid (partially available) directory state.
*   Always disable caching for exit handlers in Kubeflow V2 runner to
    reflect latest status of dependent dag.

## Documentation Updates

*   N/A

