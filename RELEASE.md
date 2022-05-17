# Version 1.8.0

## Major Features and Improvements

*   Added experimental exit_handler support for KubeflowDagRunner.
*   Enabled custom labels to be submitted to CAIP training jobs.
*   Enabled custom Python function-based components to share pipeline Beam
    configuration by [inheriting from BaseBeamComponent]
(https://www.tensorflow.org/tfx/guide/custom_function_component)

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

* `LatestBlessedModelStrategy` gracefully handles the case where there are no
  blessed model at all (e.g. first run).
* Fix that the resolver with custom `ResolverStrategy` (assume correctly
  packaged) fails.
* Fixed `ElwcBigQueryExampleGen` data serializiation error that was causing an
  assertion failure on Beam.
* Added dark mode styling support for InteractiveContext notebook formatters.
* (Python 3.9+) Supports `list` and `dict` in type definition of execution
  properties.
* Populate Artifact proto `name` field when name is set on the Artifact python
  object.
* Temporarily capped `apache-airflow` version to 2.2.x to avoid dependency
  conflict. We will rollback this change once `kfp` releases a new version.
* Fixed a compatibility issue with apache-airflow 2.3.0 that is failing with
  "unexpected keyword argument 'default_args'".
* StatisticsGen will raise an error if unsupported StatsOptions (i.e.,
  generators or experimental_slice_functions) are passed.

## Dependency Updates

| Package Name | Version Constraints | Previously (in `v1.7.0`) | Comments |
| -- | -- | -- | -- |
| `apache-beam[gcp]` | `>=2.38,<3` | `>=2.36,<3` | Synced release train |

## Documentation Updates

*   N/A

