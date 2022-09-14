# Version 1.10.0

## Major Features and Improvements

*   Saved tuner results in pandas `records` formatted JSON.
*   TFX Transform now supports `tf.SequenceExample` natively. The native path can be activated by providing `TensorRepresentation`s in the Schema.
*   TFX Transform now supports reading raw and materializing transformed data in
    Apache Parquet format.
*   ExampleDiff outputs statistics on the matching process, and optional counts
    of paired feature values.

## Breaking Changes

*   N/A

### For Pipeline Authors

*   N/A

### For Component Authors

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Type hint on BaseComponent.inputs and BaseComponent.outputs corrected to be
    Channel subclasses.
*   Added `input_optional` parameter to `ChannelParameter`. This allows
    component authors to declare that even if a channel is `optional`, if it is
    provided during pipeline definition time, then it must have resolved inputs
    during run time.
*   Allow latest `apache-airflow` 2.x versions.
*   Moved `tflite-support` related dependencies from `[examples]` to a separate
    `[tflite-support]` extra.

## Dependency Updates

| Package Name | Version Constraints | Previously (in `v1.9.0`) | Comments |
| -- | -- | -- | -- |
| `google-api-core` | `<2` | N/A | Added to help pip dependency resolution. google-api-core was already a transitive dependency. |
| `apache-beam[gcp]` | `>=2.40,<3` | `>=2.38,<3` | Synced release train |
| `attrs` | `>=19.3.0,<22` | `>=19.3.0,<21` | Allow more recent versions |
| `pyarrow` | `>=6,<7` | `>=1,<6` | Synced release train |
| `tflite-support` | `~=0.4.2` | `>=0.1.0a1,<0.2.1` | Update to a TF-2.10 compatible version. |

## Documentation Updates

*   N/A
