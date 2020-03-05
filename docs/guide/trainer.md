# The Trainer TFX Pipeline Component

The Trainer TFX pipeline component trains a TensorFlow model.

## Trainer and TensorFlow

Trainer makes extensive use of the Python
[TensorFlow](https://www.tensorflow.org) API for training models.

Note: TFX supports TensorFlow 1.15 and 2.x.

## Component

Trainer takes:

*   tf.Examples used for training and eval.
*   A user provided module file that defines the trainer logic.
*   A data schema created by a SchemaGen pipeline component and optionally
    altered by the developer.
*   (https://developers.google.com/protocol-buffers)[Protobuf]
definition of train args and eval args.
*   (Optional) transform graph produced by an upstream Transform component.
*   (Optional) pre-trained models used for scenarios such as warmstart.

Trainer emits: A SavedModel and an optional EvalSavedModel

## Estimator based Trainer

To learn about using an (https://www.tensorflow.org/guide/estimator)[Estimator]
based model with TFX and Trainer, see
[Designing TensorFlow modeling code with tf.Estimator for TFX](train.md).

### Configuring a Trainer Component

Typical pipeline Python DSL code looks like this:

```python
from tfx.components import Trainer

...

trainer = Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      base_models=latest_model_resolver.outputs['latest_model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer invokes a training module, which is specified in the `module_file`
parameter.  A typical training module looks like this:

```python
# TFX will call this function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator using the high level API.

  Args:
    trainer_fn_args: Holds args used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  tf_transform_output = tft.TFTransformOutput(trainer_fn_args.transform_output)

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      trainer_fn_args.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)
  warm_start_from = trainer_fn_args.base_models[
      0] if trainer_fn_args.base_models else None

  estimator = _build_estimator(
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=warm_start_from)

  # Create an input receiver for TFMA processing
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      tf_transform_output, schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }
```

## Generic Trainer

To better support native Keras, we proposed a generic trainer which allows any
TensorFlow training loop in TFX Trainer in addition to tf.estimator. For
details, please see the
[RFC for generic trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

### Configuring a Trainer Component with a generic executor

Typical pipeline Python DSL code looks like this:

```python
from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor

...

trainer = Trainer(
    module_file=module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer invokes a training module, which is specified in the `module_file`
parameter. Instead of `trainer_fn`, a `run_fn` is required in the module file if
`GenericExecutor` is specified. A typical training module looks like this:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    transformed_features.pop(_transformed_name(_LABEL_KEY))

    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

### [tf.distribute.Strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy)

Single worker distribution strategy is supported in TFX currently. For
multi-worker strategy, it requires the implementation of TFX Trainer in certain
execution environment has the ability to bring up the cluster of worker
machines, currently the default TFX Trainer doesn't have ability to spawn
multi-worker cluster, Cloud AI Platform support is WIP.

To use distribution strategy, create an appropriate tf.distribute.Strategy and
move the creation and compiling of Keras model inside strategy scope.

For example, replace `model = _build_keras_model()` with:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
will use CPU if no GPUs are found. To verify it actually uses GPU, enable info
level tensorflow logging:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

and you should be able to see `Using MirroredStrategy with devices (...GPUs...)`
in the log.

Note: environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true` might be needed for
GPU out of memory issue.
