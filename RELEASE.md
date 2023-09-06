# Version 1.14.0

## Major Features and Improvements

*  Added python 3.10 support.

## Breaking Changes

*   `Placeholder` (and `_PlaceholderOperator`) are no longer `Jsonable`.
*   Optimize MLMD register type to one call in most time instead of two calls.

### For Pipeline Authors

*   N/A

### For Component Authors

* Replace "tf_estimator" with "tfma_eval" as the identifier for tfma
  EvalSavedModel. "tf_estimator" is now serves as the identifier for the normal
  estimator model with any signature (by default 'serving').

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*  Apply latest TFX image vulnerability resolutions (base OS and software updates)

## Dependency Updates
| Package Name | Version Constraints | Previously (in `v1.13.0`) | Comments |
| -- | -- | -- | -- |
| `tensorflow-hub` | `>=0.9.0,<0.14` | `>=0.9.0,<0.13` | |
| `pyarrow` | `>=10,<11` | `>=6,<7` | |
| `apache-beam` | `>=2.40,<3` | `>=2.47,<3` | |
| `scikit-learn` | `>=1.0,<2` | `>=0.23,<0.24` | |
| `google-api-core` | `<3` | `<1.33` | |
| `google-cloud-aiplatform` | `>=1.6.2,<2` | `>=1.6.2,<1.18` | |
| `tflite-support` | `>=0.4.3,<0.4.5` | `>=0.4.2,<0.4.3` | |
| `pyyaml` | `>=6,<7`| `>=3.12,<6` | Issue with installation of PyYaml 5.4.1. (https://github.com/yaml/pyyaml/issues/724) |
| `tensorflow` | `>=2.13,<2.14` | `>=2.12,<2.13` | |
| `tensorflowjs` | `>=4.5,<5` | `>=3.6.0,<4` | |

## Documentation Updates

*  N/A