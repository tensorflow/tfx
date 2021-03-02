# Version 0.28.0

## Major Features and Improvements

*   Publically released TFX docker image in [tensorflow/tfx](
    https://hub.docker.com/r/tensorflow/tfx) will use GPU
    compatible based TensorFlow images from [Deep Learning Containers](
    https://cloud.google.com/ai-platform/deep-learning-containers). This allow
    these images to be used with GPU out of box.
*   Added an example pipeline for a ranking model (using
    [tensorflow_ranking](https://github.com/tensorflow/ranking))
    at `tfx/examples/ranking`. More documentation will be available in future
    releases.
*   Added a [spans_resolver](
    https://github.com/tensorflow/tfx/blob/master/tfx/dsl/experimental/spans_resolver.py)
    that can resolve spans based on range_config.

## Breaking Changes

### For Pipeline Authors

*   Custom arg key in `google_cloud_ai_platform.tuner.executor` is renamed to
    `ai_platform_tuning_args` from `ai_platform_training_args`, to better
    distinguish usage with Trainer.

### For component authors

*   N/A

## Deprecations

*   Deprecated input/output compatibility aliases for Transform and SchemaGen.

## Bug Fixes and Other Changes

*   Change Bigquery ML Pusher to publish the model to the user specified project
    instead of the default project from run time context.
*   Depends on `apache-beam[gcp]>=2.28,<3`.
*   Depends on `ml-metadata>=0.28.0,<0.29.0`.
*   Depends on `kfp-pipeline-spec>=0.1.6,<0.2`.
*   Depends on `struct2tensor>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-data-validation>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-model-analysis>=0.28.0,<0.29.0`.
*   Depends on `tensorflow-transform>=0.28.0,<0.29.0`.
*   Depends on `tfx-bsl>=0.28.1,<0.29.0`.

## Documentation Updates

*   Published a [migration instruction](
    https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/launcher/README.md)
    for legacy custom launcher developers.
