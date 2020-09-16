# Version 0.24.0

## Major Features and Improvements

*   Use TFXIO and batched extractors by default in Evaluator.
*   Supported split configuration for Transform.
*   Added python 3.8 support.

## Bug fixes and other changes

*   Supported CAIP Runtime 2.2 for online prediction pusher.
*   Stopped depending on `Werkzeug`.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `ml-metadata>=0.24,<0.25`.
*   Depends on `tensorflow-data-validation>=0.24,<0.25`.
*   Depends on `tensorflow-model-analysis>=0.24,<0.25`.
*   Depends on `tensorflow-transform>=0.24,<0.25`.
*   Depends on `tfx-bsl>=0.24,<0.25`.

## Breaking changes

*   N/A

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Documentation updates

*   N/A

## Deprecations

*   Deprecated python 3.5 support.
