# Version 1.16.0

## Major Features and Improvements

*   N/A

## Breaking Changes

*  `Placeholder.__format__()` is now disallowed, so you cannot use placeholders
   in f-strings and `str.format()` calls anymore. If you get an error from this,
   most likely you discovered a bug and should not use an f-string in the first
   place. If it is truly your intention to print the placeholder (not its
   resolved value) for debugging purposes, use `repr()` or `!r` instead.
* Drop supports for the Estimator API.

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   KubeflowDagRunner (KFP v1 SDK) is deprecated. Use KubeflowV2DagRunner (KFP v2 pipeline spec) instead.
*   Since Estimators will no longer be available in TensorFlow 2.16 and later versions, we have deprecated examples and templates that use them. We encourage you to explore Keras as a more modern and flexible high-level API for building and training models in TensorFlow.

## Bug Fixes and Other Changes

*   N/A

## Dependency Updates
| Package Name | Version Constraints | Previously (in `v1.15.1`) | Comments |
| -- | -- | -- | -- |
| `docker` | `>=7,<8` | `>=4.1,<5` | |

## Documentation Updates

*   N/A

