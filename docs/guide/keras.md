# TensorFlow 2.x in TFX

[TensorFlow 2.0 was released in 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html),
with
[tight integration of Keras](https://www.tensorflow.org/guide/keras/overview),
[eager execution](https://www.tensorflow.org/guide/eager) by default, and
[Pythonic function execution](https://www.tensorflow.org/guide/function), among
other
[new features and improvements](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

This guide provides a comprehensive technical overview of TF 2.x in TFX.

## Which version to use?

TFX is compatible with TensorFlow 2.x, and the high-level APIs that existed in
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

### Converting existing projects to TensorFlow 2.x

Code written for TensorFlow 1.x is largely compatible with TensorFlow 2.x and
will continue to work in TFX.

However, if you'd like to take advantage of improvements and new features as
they become available in TF 2.x, you can follow the
[instructions for migrating to TF 2.x](https://www.tensorflow.org/guide/migrate).

## Estimator

The Estimator API has been retained in TensorFlow 2.x, but is not the focus of
new features and development. Code written in TensorFlow 1.x or 2.x using
Estimators will continue to work as expected in TFX.

Here is an end-to-end TFX example using pure Estimator:
[Taxi example (Estimator)](https://github.com/tensorflow/tfx/blob/r0.21/tfx/examples/chicago_taxi_pipeline/taxi_utils.py)

## Keras with `model_to_estimator`

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
  """Build the estimator, using model_to_estimator."""
  ...

  # Model to estimator
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  return {
      'estimator': estimator,
      ...
  }
```

Other than the user module file of Trainer, the rest of the pipeline remains
unchanged.

## Native Keras (i.e. Keras without `model_to_estimator`)

Note: Full support for all features in Keras is in progress, in most cases,
Keras in TFX will work as expected. It does not yet work with Sparse Features
for FeatureColumns.

### Examples and Colab

Here are several examples with native Keras:

*   [Penguin](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils.py)):
    'Hello world' end-to-end example.
*   [MNIST](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_pipeline_native_keras.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/mnist/mnist_utils_native_keras.py)):
    Image and TFLite end-to-end example.
*   [Taxi](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_native_keras.py)
    ([module file](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py)):
    end-to-end example with advanced Transform usage.

We also have a per-component
[Keras Colab](https://www.tensorflow.org/tfx/tutorials/tfx/components_keras).

### TFX Components

The following sections explain how related TFX components support native Keras.

#### Transform

Transform currently has experimental support for Keras models.

The Transform component itself can be used for native Keras without change. The
`preprocessing_fn` definition remains the same, using
[TensorFlow](https://www.tensorflow.org/api_docs/python/tf) and
[tf.Transform](https://www.tensorflow.org/tfx/transform/api_docs/python/tft)
ops.

The serving function and eval function are changed for native Keras. Details
will be discussed in the following Trainer and Evaluator sections.

Note: Transformations within the `preprocessing_fn` cannot be applied to the
label feature for training or eval.

#### Trainer

To configure native Keras, the `GenericExecutor` needs to be set for Trainer
component to replace the default Estimator based executor. For details, please
check
[here](trainer.md#configuring-the-trainer-component-to-use-the-genericexecutor).

##### Keras Module file with Transform

The training module file must contains a `run_fn` which will be called by the
`GenericExecutor`, a typical Keras `run_fn` would look like this:

```python
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  # Train and eval files contains transformed examples.
  # _input_fn read dataset based on transformed schema from tft.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output.transformed_metadata.schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                           tf_transform_output.transformed_metadata.schema)

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

In the `run_fn` above, a serving signature is needed when exporting the trained
model so that model can take raw examples for prediction. A typical serving
function would look like this:

```python
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  # the layer is added as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting.
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
```

In above serving function, tf.Transform transformations need to be applied to
the raw data for inference, using the
[`tft.TransformFeaturesLayer`](https://github.com/tensorflow/transform/blob/master/docs/api_docs/python/tft/TransformFeaturesLayer.md)
layer. The previous `_serving_input_receiver_fn` which was required for
Estimators will no longer be needed with Keras.

##### Keras Module file without Transform

This is similar to the module file shown above, but without the transformations:

```python
def _get_serve_tf_examples_fn(model, schema):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = _get_raw_feature_spec(schema)
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn


def run_fn(fn_args: TrainerFnArgs):
  schema = io_utils.parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())

  # Train and eval files contains raw examples.
  # _input_fn reads the dataset based on raw data schema.
  train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor, schema)
  eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor, schema)

  model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, schema).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
```

##### [tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)

At this time TFX only supports single worker strategies (e.g.,
[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy),
[OneDeviceStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/OneDeviceStrategy)).

To use a distribution strategy, create an appropriate tf.distribute.Strategy and
move the creation and compiling of the Keras model inside a strategy scope.

For example, replace above `model = _build_keras_model()` with:

```python
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  # Rest of the code can be unchanged.
  model.fit(...)
```

To verify the device (CPU/GPU) used by `MirroredStrategy`, enable info level
tensorflow logging:

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)
```

and you should be able to see `Using MirroredStrategy with devices (...)` in the
log.

Note: The environment variable `TF_FORCE_GPU_ALLOW_GROWTH=true` might be needed
for a GPU out of memory issue. For details, please refer to
[tensorflow GPU guide](https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth).

#### Evaluator

In TFMA v0.2x, ModelValidator and Evaluator have been combined into a single
[new Evaluator component](https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-combining-model-validator-with-evaluator.md).
The new Evaluator component can perform both single model evaluation and also
validate the current model compared with previous models. With this change, the
Pusher component now consumes a blessing result from Evaluator instead of
ModelValidator.

The new Evaluator supports Keras models as well as Estimator models. The
`_eval_input_receiver_fn` and eval saved model which were required previously
will no longer be needed with Keras, since Evaluator is now based on the same
`SavedModel` that is used for serving.

[See Evaluator for more information](evaluator.md).
