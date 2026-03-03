# Version 1.17.0

## Major Features and Improvements

*   N/A

## Breaking Changes

*   N/A

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Ensured `tensorflow-data-validation` and `tfx-bsl` are built from
      source within the Docker image instead of PYPI to prevent wheels Incompatibility.
*   Updated end-to-end tests for Kubeflow v2 to use hardcoded container
      paths for test data, resolving `RuntimeError` issues.
*   Introduced patches for `tfdv`, `tfx-bsl`, and `tfx` to facilitate
      compilation issues and to resolve dependency conflicts.

## Dependency Updates

*   Supports Tensorflow | `>=2.17.0,<2.18.0` |
*   supports Protobuf  | `==4.21.12` |

## Documentation Updates

*   N/A
