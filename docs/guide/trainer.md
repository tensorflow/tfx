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
*   [Protobuf](https://developers.google.com/protocol-buffers) definition of
    train args and eval args.
*   (Optional) A data schema created by a SchemaGen pipeline component and
    optionally altered by the developer.
*   (Optional) transform graph produced by an upstream Transform component.
*   (Optional) pre-trained models used for scenarios such as warmstart.
*   (Optional) hyperparameters, which will be passed to user module function.
    Details of the integration with Tuner can be found [here](tuner.md).

Trainer emits: At least one model for inference/serving (typically in SavedModelFormat) and optionally another model for eval (typically an EvalSavedModel).

We provide support for alternate model formats such as
[TFLite](https://www.tensorflow.org/lite) through the [Model Rewriting Library](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/rewriting/README.md).
See the link to the Model Rewriting Library for examples of how to convert both Estimator and Keras
models.

## Generic Trainer

Generic trainer enables developers to use any TensorFlow model API with the
Trainer component. In addition to TensorFlow Estimators, developers can use
Keras models or custom training loops. For details, please see the
[RFC for generic trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

### Configuring the Trainer Component

Typical pipeline DSL code for the generic Trainer would look like this:

```python
from tfx.components import Trainer

...

trainer = Trainer(
    module_file=module_file,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

Trainer invokes a training module, which is specified in the `module_file`
parameter. Instead of `trainer_fn`, a `run_fn` is required in the module file if
the `GenericExecutor` is specified in the `custom_executor_spec`. The
`trainer_fn` was responsible for creating the model. In addition to that,
`run_fn` also needs to handle the training part and output the trained model to
a the desired location given by
[FnArgs](https://github.com/tensorflow/tfx/blob/master/tfx/components/trainer/fn_args_utils.py):

```python
from tfx.components.trainer.fn_args_utils import FnArgs

def run_fn(fn_args: FnArgs) -> None:
  """Build the TF model and train it."""
  model = _build_keras_model()
  model.fit(...)
  # Save model to fn_args.serving_model_dir.
  model.save(fn_args.serving_model_dir, ...)
```

Here is an
[example module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)
with `run_fn`.

Note that if the Transform component is not used in the pipeline, then the
Trainer would take the examples from ExampleGen directly:

```python
trainer = Trainer(
    module_file=module_file,
    examples=example_gen.outputs['examples'],
    schema=infer_schema.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(num_steps=10000),
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))
```

More details are available in the
[Trainer API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Trainer).
