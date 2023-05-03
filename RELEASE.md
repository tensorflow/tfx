# Version 1.13.0

## Major Features and Improvements

*  Supported setting the container image at a component level for Kubeflow V2
   Dag Runner.

## Breaking Changes

### For Pipeline Authors

*   Conditional can be used from `tfx.dsl.Cond` (Given `from tfx import v1 as
    tfx`).
*   Dummy channel for testing can be constructed by
    `tfx.testing.Channel(artifact_type)`.
*   `placeholder.Placeholder.placeholders_involved()` was replaced with
    `placeholder.Placeholder.traverse()`.
*   `placeholder.Predicate.dependent_channels()` was replaced with
    `channel_utils.get_dependent_channels(Placeholder)`.
*   `placeholder.Predicate.encode_with_keys(...)` was replaced with
    `channel_utils.encode_placeholder_with_channels(Placeholder, ...)`.

*   `placeholder.Predicate.from_comparison()` removed (was deprecated)
*   enable `external_pipeline_artifact_query` for querying artifact within one pipeline
*   Support `InputArtifact[List[Artifact]]` annotation in Python function custom component

### For Component Authors

*   N/A

## Deprecations

*   Deprecate python 3.7 support

## Bug Fixes and Other Changes

*  Support to task type "workerpool1" of CLUSTER_SPEC in Vertex AI training's 
   service according to the changes of task type in Tuner component.
*  Propagates unexpected import failures in the public v1 module.

## Dependency Updates
| Package Name | Version Constraints | Previously (in `v1.12.0`) | Comments |
| -- | -- | -- | -- |
| `click` | `>=7,<9` | `>=7,<8` | |
| `ml-metadata` | `~=1.13.1` | `~=1.12.0` | Synced release train |
| `protobuf` | `>=3.13,<4` | `>=3.20.3,<5` | To support TF 2.12|
| `struct2tensor` | `~=0.44.0` | `~=0.43.0` | Synced release train |
| `tensorflow` | `~=2.12.0` | `>=1.15.5,<2` or `~=2.11.0` | |
| `tensorflow-data-validation` | `~=1.13.0` | `~=1.12.0` | Synced release train |
| `tensorflow-model-analysis` | `~=0.44.0` | `~=0.43.0` | Synced release train |
| `tensorflow-transform` | `~=1.13.0` | `~=1.12.0` | Synced release train |
| `tfx-bsl` | `~=1.13.0` | `~=1.12.0` | Synced release train |

## Documentation Updates

*  Added page for TFX-Addons