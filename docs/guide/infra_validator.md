# The InfraValidator TFX Pipeline Component

InfraValidator is a TFX component that is used as an early warning layer before
pushing a model into production. The name "infra" validator came from the fact
that it is validating the model in the actual model serving "infrastructure". If
[evaluator](https://www.tensorflow.org/tfx/guide/evaluator) is to guarantee the
performance of the model, InfraValidator is to guarantee the model is
mechanically fine and prevents bad models from being pushed.

## How does it work?

InfraValidator takes the model, launches a sand-boxed model server with the
model, and sees if it can be successfully loaded and optionally queried. The
infra validation result will be generated in the `blessing` output in the same
way as [evaluator](https://www.tensorflow.org/tfx/guide/evaluator) does.

InfraValidator focuses on the compatibility between the model server binary
(e.g. [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)) and
the model to deploy. Despite the name "infra" validator, it is the **user's
responsibility** to configure the environment correctly, and infra validator
only interacts with the model server in the user-configured environment to see
if it works fine. Configuring this environment correctly will ensure that infra
validation passing or failing will be indicative of whether the model would be
servable in the production serving environment. This implies some of, not
limited to, the following:

1.  InfraValidator is using the same model configurations including the assets
    (e.g. vocabulary dictionary) for model as will be used in production.
2.  InfraValidator is using the same model server binary as will be used in
    production. This is the minimal level to which the infra validation
    environment must converge.
3.  InfraValidator is using the same resources (e.g. allocation quantity and
    type of CPU, memory, and accelerators) as will be used in production.
4.  InfraValidator is using the same model server configuration as will be used
    in production.

Depending on the situation, users can choose to what degree InfraValidator
should be identical to the production environment. Technically, a model can be
infra validated in a local Docker environment and then served in a completely
different environment (e.g. Kubernetes cluster) without a problem. However,
InfraValidator will not have checked for this divergence.

### Operation mode

Depending on the configuration, infra validation is done in one of the following
modes:

-   `LOAD_ONLY` mode: checking whether the model was successfully loaded in the
    serving infrastructure or not. **OR**
-   `LOAD_AND_QUERY` mode: `LOAD_ONLY` mode plus sending some sample requests to
    check if model is capable of serving inferences. InfraValidator does not
    care the prediction was correct or not. Only whether the request was
    successful or not matters.

## How do I use it?

Usually InfraValidator is defined next to an Evaluator component, and its output
is fed to a Pusher. If InfraValidator fails, the model will not be pushed.

```python {highlight="lines:8-11 context:infra_blessing,1"}
evaluator = Evaluator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=EvalConfig(...)
)

infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=ServingSpec(...)
)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=PushDestination(...)
)
```

### Configuring an InfraValidator component.

There are three kinds of protos to configure InfraValidator.

#### `ServingSpec`

`ServingSpec` is the most crucial configuration for the InfraValidator. It
defines:

-   <u>what</u> type of model server to run
-   <u>where</u> to run it

For model server types (called serving binary) we support

-   [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)

Note: InfraValidator allows specifying multiple versions of the same model
server type in order to upgrade the model server version without affecting model
compatibility. For example, user can test `tensorflow/serving` image with both
`2.1.0` and `latest` versions, to ensure the model will be compatible with the
latest `tensorflow/serving` version as well.

Following serving platforms are currently supported:

-   Local Docker (Docker should be installed in advance)
-   Kubernetes (limited support for KubeflowDagRunner only)

The choice for serving binary and serving platform are made by specifying a
[`oneof`](https://developers.google.com/protocol-buffers/docs/proto3#oneof)
block of the `ServingSpec`. For example to use TensorFlow Serving binary running
on the Kubernetes cluster, `tensorflow_serving` and `kubernetes` field should be
set.

```python {highlight="lines:4:9-4:26,7:9-7:18"}
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=ServingSpec(
        tensorflow_serving=TensorFlowServing(
            tags=['latest']
        ),
        kubernetes=KubernetesConfig()
    )
)
```

To further configure `ServingSpec`, please check out the
[protobuf definition](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `ValidationSpec`

Optional configuration to adjust the infra validation criteria or workflow.

```python {highlight="lines:4-10"}
infra_validator=InfraValidator(
    model=trainer.outputs['model'],
    serving_spec=ServingSpec(...),
    validation_spec=ValidationSpec(
        # How much time to wait for model to load before automatically making
        # validation fail.
        max_loading_time_seconds=60,
        # How many times to retry if infra validation fails.
        num_tries=3
    )
)
```

All ValidationSpec fields have a sound default value. Check more detail from the
[protobuf definition](https://github.com/tensorflow/tfx/blob/master/tfx/proto/infra_validator.proto).

#### `RequestSpec`

Optional configuration to specify how to build sample requests when running
infra validation in `LOAD_AND_QUERY` mode. In order to use `LOAD_AND_QUERY`
mode, it is required to specify both `request_spec` execution properties as well
as `examples` input channel in the component definition.

```python {highlight="lines:7:9-7:62 lines:10-16"}
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    # This is the source for the data that will be used to build a request.
    examples=example_gen.outputs['examples'],
    serving_spec=ServingSpec(
        # Depending on what kind of model server you're using, RequestSpec
        # should specify the compatible one.
        tensorflow_serving=TensorFlowServing(tags=['latest']),
        local_docker=LocalDockerConfig(),
    ),
    request_spec=RequestSpec(
        # This is the recipe to build a request from `examples` input artifact.
        tensorflow_serving=TensorFlowServingRequestSpec(
            signature_names=['classification']
        ),
        num_examples=10  # How many requests to make.
    )
)
```

## Limitations

Current InfraValidator has some limitations on a few features it provides.

-   For Kubernetes serving platform, the pipeline should be executed by
    `KubeflowDagRunner` inside Kubeflow Pipelines. The model server will be
    launched in the same Kubernetes cluster and the namespace that Kubeflow is
    using.
-   There's a limited support on `LOAD_AND_QUERY` mode for the
    [Predict](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model/predict_signature_def)
    method signature (which is the only exportable method in TensorFlow 2).
    InfraValidator requires the Predict signature to consume a serialized
    [`tf.Example`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample)
    as the only input.

    ```python
    @tf.function
    def parse_and_run(serialized_example):
      features = tf.io.parse_example(serialized_example, FEATURES)
      return model(features)

    model.save('path/to/save', signatures={
      # This exports "Predict" method signature under name "serving_default".
      'serving_default': parse_and_run.get_concrete_function(
          tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    })
    ```

    Check out an
    [Iris example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)
    sample code to see how this signature interacts with other components in
    TFX.
