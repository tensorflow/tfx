# Version 0.26.0

## Major Features and Improvements

*   Supported output examples artifact for BulkInferrer which can be used to
    link with downstream training.
*   TFX Transform switched to a (notably) faster and more accurate
    implementation of `tft.quantiles` analyzer.
*   Added native TF 2 implementation of Transform. The default
    behavior will continue to use Tensorflow's compat.v1 APIs. This can be
    overriden by passing `force_tf_compat_v1=False` and enabling TF 2 behaviors.
    The default behavior for TF 2 will be switched to the new native
    implementation in a future release.
*   Added support for passing a callable to set pre/post transform statistic
    generation options.
*   In addition to the "tfx" pip package, a dependency-light distribution of the
    core pipeline authoring functionality of TFX is now available as the
    "ml-pipelines-sdk" pip package. This package does not include first-party
    TFX components. The "tfx" pip package is still the recommended installation
    path for TFX.

## Breaking changes

*   Wheel package building for TFX has changed, and users need to follow the
    [new TFX package build instructions]
    (https://github.com/tensorflow/tfx/blob/master/package_build/README.md) to
    build wheels for TFX.

### For pipeline authors

*   N/A

### For component authors

*   N/A

## Deprecations

*   TrainerFnArgs is deprecated by FnArgs.
*   Deprecated DockerComponentConfig class: user should set a DockerPlatformConfig
    proto in `platform_config` using `with_platform_config()` API instead.

## Bug fixes and other changes

*   Official TFX container image's entrypoint is changed so the image can be
    used as a custom worker for Dataflow.
*   In the published TFX container image, wheel files are now used to install
    TFX, and the TFX source code has been moved to `/tfx/src`.
*   Added a skeleton of CLI support for Kubeflow V2 runner, and implemented
    support for pipeline operations.
*   Added an experimental template to use with Kubeflow V2 runner.
*   Added sanitization of user-specified pipeline name in Kubeflow V2 runner.
*   Migrated `deployment_config` in Kubeflow V2 runner from `Any` proto message
    to `Struct`, to ensure compatibility across different copies of the proto
    libraries.
*   The `tfx.dsl.io.fileio` filesystem handler will delegate to
    `tensorflow.io.gfile` for any unknown filesystem schemes if TensorFlow
    is installed.
*   Skipped ephemeral package when the beam flag
    'worker_harness_container_image' is set.
*   The `tfx.dsl.io.makedirs` call now succeeds if the directory already exists.
*   Depends on `apache-beam[gcp]>=2.25,!=2.26,<3`.
*   Depends on `keras-tuner>=1,<1.0.2`.
*   Depends on `kfp-pipeline-spec>=0.1.3,<0.2`.
*   Depends on `ml-metadata>=0.26.0,<0.27.0`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-data-validation>=0.26,<0.27`.
*   Depends on `tensorflow-model-analysis>=0.26,<0.27`.
*   Depends on `tensorflow-serving>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-transform>=0.26,<0.27`.
*   Depends on `tfx-bsl>=0.26.1,<0.27`.

## Documentation updates

*   N/A
