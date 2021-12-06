# Version 1.5.0

## Major Features and Improvements

*   Added support for partial pipeline run. Users can now run a subset of nodes
    in a pipeline while reusing artifacts generated in previous pipeline runs.
    This is supported in LocalDagRunner and BeamDagRunner, and is exposed via
    the TfxRunner API.

## Breaking Changes

*   N/A

### For Pipeline Authors

*   N/A

### For Component Authors

*   N/A

## Deprecations

*   N/A

## Bug Fixes and Other Changes
*   Increased docker timeout to 5 minutes for image building in CLI.
*   Fixed KeyError when multiple Examples artifacts were used in Transform
    without materialization.
*   Fixed error where Vertex Endpoints of the same name is not deduped
*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<2.8`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `ml-metadata>=1.5.0,<1.6.0`.
*   Depends on `struct2tensor>=0.36.0,<0.37.0`.
*   Depends on `tensorflow-data-validation>=1.5.0,<1.6.0`.
*   Depends on `tensorflow-model-analysis>=0.36.0,<0.37.0`.
*   Depends on `tensorflow-transform>=1.5.0,<1.6.0`.
*   Depends on `tfx-bsl>=1.5.0,<1.6.0`.

## Documentation Updates

*   N/A
