# Keras and TF 2.0 in TFX

[TensorFlow 2.0 was released in 2019](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html),
with
[tight integration of Keras](https://www.tensorflow.org/guide/keras/overview),
[eager execution](https://www.tensorflow.org/guide/eager) by default, and
[Pythonic function execution](https://www.tensorflow.org/guide/function), among
other
[new features and improvements](https://www.tensorflow.org/guide/effective_tf2#a_brief_summary_of_major_changes).

TFX is compatible with TensorFlow 2.x, but complete support is ongoing. This
guide covers what works, what doesn't work yet, and how to work effectively with
TensorFlow 2.x in TFX.

## Start new projects in TF 2

If you are starting a new TFX project, we recommend that you use TensorFlow 2.

## Convert existing projects to TF 2

Code written for TensorFlow 1.x is largely compatible with TensorFlow 2. See
[this guide for migrating to TensorFlow 2.0](https://www.tensorflow.org/guide/migrate).

## Keras

At this time:

-   Keras layers do not work in Transform.
-   Keras models do not work with Distributed Strategy.
-   Keras works with Trainer and Model Analysis, using
    [model_to_estimator](https://www.tensorflow.org/tutorials/estimator/keras_model_to_estimator).

### Keras models in TFX Trainer

The [TFX Trainer component](https://www.tensorflow.org/tfx/guide/trainer) was
built to support the
[Estimator API](https://www.tensorflow.org/guide/estimator), and does not yet
support the [Keras Model API](https://www.tensorflow.org/guide/keras/overview).
Therefore, to use Keras models you must wrap them in with
`tf.keras.estimator.model_to_estimator`.

For example:

```py

# Build a Keras model.
def _keras_model_builder():
  """Creates a DNN Keras model  for classifying data.
  Returns:
    A keras Model.
  """

  l = tf.keras.layers
  opt = tf.keras.optimizers
  inputs = [l.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  input_layer = l.concatenate(inputs)
  d1 = l.Dense(8, activation='relu')(input_layer)
  output = l.Dense(3, activation='softmax')(d1)
  model = tf.keras.Model(inputs=inputs, outputs=output)
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=opt.Adam(lr=0.001),
      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
  absl.logging.info(model.summary())
  return model


# Write a typical trainer function
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

  train_batch_size = 40
  eval_batch_size = 40

  train_input_fn = lambda: _input_fn(
      trainer_fn_args.train_files,
      schema,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(
      trainer_fn_args.eval_files,
      schema,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=trainer_fn_args.train_steps)

  serving_receiver_fn = lambda: _serving_input_receiver_fn(schema)

  exporter = tf.estimator.FinalExporter('iris', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=trainer_fn_args.eval_steps,
      exporters=[exporter],
      name='iris-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=trainer_fn_args.serving_model_dir)

  # MODEL TO ESTIMATOR
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=_keras_model_builder(), config=run_config)

  # Create an input receiver for TFMA processing
  eval_receiver_fn = lambda: _eval_input_receiver_fn(schema)

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': eval_receiver_fn
  }
```
