<!-- TODO(jxzheng): Add other sections -->

# Kubeflow Pipelines V2 Runner

This is the runner compiling TFX pipeline to
[KFP Pipeline IR](https://github.com/kubeflow/pipelines/blob/master/api/v2alpha1/pipeline_spec.proto).
**Note**: This is different than the current
[TFX pipeline IR](https://github.com/tensorflow/tfx/blob/master/tfx/proto/orchestration/pipeline.proto).

## SDK
In order to use this library, users need `kfp`, whose pipeline spec is required.
One can get that by installing `tfx` with the `test` dependencies:
```
pip install tfx[test]
```

### Current support for TFX components
In current phase, the V2 runner supports most first party
components as well as custom components.

#### TFX first party components
Following components have been validated and officially support:

* `BulkInferrer`
* `Evaluator`
* `ExampleGen`. Users are not able to customize the driver logic for
  `FileBasedExampleGen` yet.
* `ExampleValidator`
* `Importer`
* `Resolver`. Currently managed pipeline support two resolver policies:
  [latest blessed model](https://github.com/tensorflow/tfx/blob/dd6a38120e5699428eac143f762a1d5b3442b239/tfx/dsl/experimental/latest_blessed_model_resolver.py#L32)
   and [latest artifact](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/experimental/latest_artifacts_resolver.py#L30)
   if the artifact type belongs to `standard_artifacts`.
* `Pusher`
* `SchemaGen`
* `StatisticsGen`
* `Trainer`
* `Transform`

#### Experimental feature
* `AiPlatformTrainingComponent`. Users can use
  `tfx.orchestration.kubeflow.v2.components.experimental.ai_platform_training_component.create_ai_platform_training`
  to create a TFX component to launch arbitrary custom
  container training job on Cloud AI Platform Training, with an easy-to-use
  component interface.

#### Custom components
Users are also able to bring in custom components by following
[the custom component guide](https://www.tensorflow.org/tfx/guide/custom_component).
More examples can be found [here](https://github.com/tensorflow/tfx/tree/eb0298462b0fd0b346e964f043e7263f39a3cddf/tfx/examples/custom_components).
