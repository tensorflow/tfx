# Version 0.30.1

## Major Features and Improvements

*   TFX CLI now supports
    [Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/introduction).
    use it with `--engine=vertex` flag.

## Breaking Changes

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes

*   Fix resolver artifact filter in TFX -> KFP IR compiler with OP filter syntax.
*   Forces keyword arguments for AirflowComponent to make it compatible with
    Apache Airflow 2.1.0 and later.

## Documentation Updates

*   N/A
