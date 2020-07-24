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
*   [Protobuf](https://developers.google.com/protocol-buffers) definition of
    train args and eval args.
*   (Optional) transform graph produced by an upstream Transform component.
*   (Optional) pre-trained models used for scenarios such as warmstart.
*   (Optional) hyperparameters, which will be passed to user module function.
    Details of the integration with Tuner can be found [here](tuner.md).

Trainer emits: At least one model for inference/serving (typically in SavedModelFormat) and optionally another model for eval (typically an EvalSavedModel).

We provide support for alternate model formats such as
[TFLite](https://www.tensorflow.org/lite) through the [Model Rewriting Library](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md).
See the link to the Model Rewriting Library for examples of how to convert both Estimator and Keras
models.


## Estimator based Trainer

To learn about using an [Estimator](https://www.tensorflow.org/guide/estimator)
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

Generic trainer enables developers to use any TensorFlow model API with the
Trainer component. In addition to TensorFlow Estimators, developers can use
Keras models or custom training loops.
For details, please see the
[RFC for generic trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

### Configuring the Trainer Component to use the GenericExecutor

Typical pipeline DSL code for the generic Trainer would look like this:

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
the `GenericExecutor` is specified in the `custom_executor_spec`.

If the Transform component is not used in the pipeline, then the Trainer would
take the examples from ExampleGen directly:

```python
trainer = Trainer(
    module_file=module_file,
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Here is an
[example module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)
with `run_fn`.
