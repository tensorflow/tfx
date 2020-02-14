# TensorFlow 2.x in TFX

[TensorFlow 2.0 was released in 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html),
with
[tight integration of Keras](https://www.tensorflow.org/guide/keras/overview),
[eager execution](https://www.tensorflow.org/guide/eager) by default, and
[Pythonic function execution](https://www.tensorflow.org/guide/function), among
other
[new features and improvements](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

This guide covers what works, what doesn't work yet, and how to work effectively
with TensorFlow 2.x in TFX.

## Which version to use?

TFX is compatible with TensorFlow 2.x., and the high-level APIs that existed in
TensorFlow 1.x (particularly Estimators) continue to work.

### Start new projects in TensorFlow 2.x

Since TensorFlow 2.x retains the high-level capabilities of TensorFlow 1.x,
there is no advantage to using the older version on new projects, even if you
don't plan to use the new features.

Therefore, if you are starting a new TFX project, we recommend that you use
TensorFlow 2.x. You may want to update your code later as full support for Keras
and other new features become available, and the scope of changes will be much
more limited if you start with TensorFlow 2.x, rather than trying to upgrade
from TensorFlow 1.x in the future.

### Consider converting existing projects to TensorFlow 2.x

Code written for TensorFlow 1.x is largely compatible with TensorFlow 2.x. and
will continue to work in TFX.

However, in order to take advantage of improvements and new features as they
become available, consider moving existing projects to TensorFlow 2.x.

For more details, see
[this guide for migrating to TensorFlow 2.0](https://www.tensorflow.org/guide/migrate).

## Keras and Estimator: Which API to use?

### Estimator

The Estimator API has been retained in TensorFlow 2.x, but is not the focus of
new features and development. Code written in TensorFlow 1.x or 2.x using
Estimators will continue to work as expected in TFX.

[Taxi example with Estimator](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

### Keras

The Keras API is the recommended way of building new models in TensorFlow 2.x.
Currently there are two different ways to move forward with Keras in TFX.

Note: Full support for all features is in progress, in most cases, keras in TFX
will work as expected. It may not work with Feature Columns and Sparse Features.

#### Model to Estimator

Keras models can be wrapped with the `tf.keras.estimator.model_to_estimator`
function, which allows them to work as if they were Estimators. To use this:

1.  Build a Keras model.
2.  Pass the compiled model into `model_to_estimator`.
3.  Use the result of `model_to_estimator` in Trainer, the way you would
    typically use an Estimator.

```py
# Build a Keras model.
def _keras_model_builder():
  """Creates a Keras model."""
  ...

  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile()

  return model


# Write a typical trainer function
def trainer_fn(trainer_fn_args, schema):
  """Build the estimator, using model_to_estimator
  """

  .
  .
  .

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': eval_receiver_fn
  }
```

[Iris example with model_to_estimator](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/iris/iris_utils.py)

#### Native Keras

To better support native Keras we proposed generic trainer, which allows any
TensorFlow training loop in TFX Trainer in addition to tf.estimator. For
details, please see the
[RFC for generic trainer](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md).

To configure native Keras, `GenericExecutor` needs to be set for Trainer
component to replace the default estimator based executor. Trainer invokes a
training module, which is specified in the `module_file` parameter. Instead of
`trainer_fn`, a `run_fn` is required in the module file if `GenericExecutor` is
specified. For details, please check [here](trainer.md).

[Iris example with native keras](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_utils_native_keras.py)
