# Version 0.27.0

## Major Features and Improvements

*   Supports different types of quantizations on TFLite conversion using
    TFLITE_REWRITER by setting `quantization_optimizations`,
    `quantization_supported_types` and `quantization_enable_full_integer`. Flag
    definitions can be found here: [Post-traning
    quantization](https://www.tensorflow.org/lite/performance/post_training_quantization).
*   Added automatic population of `tfdv.StatsOptions.vocab_paths` when computing
    statistics within the Transform component.

## Breaking changes

### For pipeline authors

*   `enable_quantization` from TFLITE_REWRITER is removed and setting
    `quantization_optimizations = [tf.lite.Optimize.DEFAULT]` will perform the
    same type of quantization, dynamic range quantization. Users of the
    TFLITE_REWRITER who do not enable quantization should be uneffected.
*   Default value for `infer_feature_shape` for SchemaGen changed from `False`
    to `True`, as indicated in previous release log. The inferred schema might
    change if you do not specify `infer_feature_shape`. It might leads to
    changes of the type of input features in Transform and Trainer code.

### For component authors

*   N/A

## Deprecations

*   Pipeline information is not be stored on the local filesystem anymore using
    Kubeflow Pipelines orchestration with CLI. Instead, CLI will always use the
    latest version of the pipeline in the Kubeflow Pipeline cluster. All
    operations will be executed based on the information on the Kubeflow
    Pipeline cluster. There might be some left files on
    `${HOME}/tfx/kubeflow` or `${HOME}/kubeflow` but those will not be used
    any more.
*   The `tfx.components.common_nodes.importer_node.ImporterNode` class has been
    moved to `tfx.dsl.components.common.importer.Importer`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   The `tfx.components.common_nodes.resolver_node.ResolverNode` class has been
    moved to `tfx.dsl.components.common.resolver.Resolver`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   The `tfx.dsl.resolvers.BaseResolver` class has been
    moved to `tfx.dsl.components.common.resolver.ResolverStrategy`, with its
    old module path kept as a deprecated alias, which will be removed in a
    future version.
*   Deprecated input/output compatibility aliases for Pusher.
*   Deprecated input/output compatibility aliases for ExampleValidator,
    Evaluator and Trainer.

## Bug fixes and other changes

*   InfraValidator supports using alternative TensorFlow Serving image in case
    deployed environment cannot reach the public internet (nor the docker hub).
    Such alternative image should behave the same as official
    `tensorflow/serving` image such as the same model volume path, serving port,
    etc.
*   Executor in `tfx.extensions.google_cloud_ai_platform.pusher.executor`
    supported regional endpoint and machine_type.
*   Starting from this version, proto files which are used to generate
    component-level configs are included in the `tfx` package directly.
*   The `tfx.dsl.io.fileio.NotFoundError` exception unifies handling of not-
    found errors across different filesystem plugin backends.
*   Fixes the serialization of zero-valued default when using `RuntimeParameter`
    on Kubeflow.
*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `ml-metadata>=0.27.0,<0.28.0`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-data-validation>=0.27.0,<0.28.0`.
*   Depends on `tensorflow-model-analysis>=0.27.0,<0.28.0`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-transform>=0.27.0,<0.28.0`.
*   Depends on `tfx-bsl>=0.27.0,<0.28.0`.

## Documentation updates

*   N/A
