# Version 1.11.0

## Major Features and Improvements

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.

*  Artifact/Channel properties now support the new MLMD PROTO property type.

*  Supports environment variables in the placeholder expression.
   This placeholder can be used to generate beam_pipeline_args
   dynamically.

## Breaking Changes

*   N/A

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Moved `tflite-support` related dependencies from `[examples]` to a separate
    `[tflite-support]` extra.
*   Moved `flax` related dependencies from `[examples]` to a separate `[flax]`
    extra.
*   Statistics gen and Schema gen now crash on empty input examples and statistics respectively.
*   Importer will now check that an existing artifact has the same type as the intended output before reusing the existing artifact.
*   Importer will now use the most recently created artifact when reusing an existing artifact instead of the one with the highest ID.
*   Proto placeholder now works with proto files that have non-trivial transitive dependencies.
*   Adding tutorials for recommenders and ranking

## Dependency Updates

| Package Name | Version Constraints | Previously (in `v1.8.0`) | Comments |
| -- | -- | -- | -- |
| `tensorflow` | `>=1.15.5,<2` or `~=2.10.0` | `>=1.15.5,<2` or `~=2.9.0` | |
| `tflite-support` | `~=0.4.2` | `>=0.1.0a1,<0.2.1` | Update to a TF-2.10 compatible version. |
| `google-cloud-aiplatform` | `>=1.6.2,<1.18` | `>=1.6.2,<2` | Added to help pip dependency resolution. |
| `ml-metadata` | `~=1.11.0` | `~=1.10.0` | Synced release train |
| `struct2tensor` | `~=0.42.0` | `~=0.41.0` | Synced release train |
| `tensorflow-data-validation` | `~=1.11.0` | `~=1.10.0` | Synced release train |
| `tensorflow-model-analysis` | `~=0.42.0` | `~=0.41.0` | Synced release train |
| `tensorflow-transform` | `~=1.11.0` | `~=1.10.0` | Synced release train |
| `tfx-bsl` | `~=1.11.0` | `~=1.10.0` | Synced release train |

## Documentation Updates

*   N/A

