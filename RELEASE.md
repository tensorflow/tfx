# Version 1.7.0

## Major Features and Improvements

* Added support for list-type Placeholder.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   Removed the already-deprecated components.ImporterNode, should use
    v1.dsl.Importer instead.
*   Deprecated Channel property setters. Use constructor argument instead.

## Bug Fixes and Other Changes

*   Fixed the cluster spec error in CAIP Tuner on Vertex when
    `num_parallel_trials = 1`
*   Replaced deprecated assertDictContainsSubset with
    assertLessEqual(itemsA, itemsB).
*   Updating Keras tutorial to make better use of Keras, and better feature
    engineering.
*   Merges KFP UI Metadata file if it already exists. Now components can produce
    their own UI results and it will be merged with existing visualization.
*   Switch Transform component to always use sketch when computing top-k stats.

## Dependency Updates

| Package Name | Version Constraints | Previously (in `v1.6.0`) | Comments |
| -- | -- | -- | -- |
| `apache-beam[gcp]` | `~=2.36` | `~=2.35` | Synced release train |
| `google-cloud-aiplatform` | `>=1.6.2,<2` | `>=1.5.0,<2` | |
| `ml-metadata` | `~=1.7.0` | `~=1.6.0` | Synced release train |
| `struct2tensor` | `~=0.38.0` | `~=0.37.0` | Synced release train |
| `tensorflow` | `>=1.15.5,<2` or `~=2.8.0` | `>=1.15.5,<2` or `~=2.7.0` | |
| `tensorflow-data-validation` | `~=1.7.0` | `~=1.6.0` | Synced release train |
| `tensorflow-decision-forests` | `==0.2.4` | `==0.2.1` | |
| `tensorflow-model-analysis` | `~=0.38.0` | `~=0.37.0` | Synced release train |
| `tensorflow-serving-api` | `>=1.15,<3` or `~=2.8.0` | `>=1.15,<3` or `~=2.7.0` | |
| `tensorflow-transform` | `~=1.7.0` | `~=1.6.0` | Synced release train |
| `tfx-bsl` | `~=1.7.0` | `~=1.6.0` | Synced release train |

## Documentation Updates

*   N/A

# Version 1.6.1

## Major Features and Improvements

*   N/A

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Fixed `Pusher` issue that didn't copy files other than
    `saved_model.pb`.

## Documentation Updates

*   N/A

