# The Trainer TFX Pipeline Component

The Trainer TFX pipeline component trains a TensorFlow model.

Trainer consumes:

* Training tf.Examples transformed by a Transform pipeline component.
* Eval tf.Examples transformed by a Transform pipeline component.
* A data schema create by a SchemaGen pipeline component and optionally altered by
the developer.

Trainer emits: A SavedModel and an EvalSavedModel

## Trainer and TensorFlow

Trainer makes extensive use of the Python
[TensorFlow](https://www.tensorflow.org) API for training models.

## Configuring a Trainer Component

A Trainer pipeline component is typically very easy to develop and requires little
customization, since all of the work is done by the Trainer TFX component.  Your
TensorFlow modeling code however may be arbitrarily complex.

Caution: Developers are strongly encouraged to use the Estimator API at this
time.  In a later release we expect Keras to be much better supported than it
currently is.

Typical code looks like this:

```python
from tfx import components

...

trainer = components.Trainer(
      module_file=taxi_pipeline_utils,
      train_files=transform_training.outputs['output'],
      eval_files=transform_eval.outputs['output'],
      schema=infer_schema.outputs['output'],
      tf_transform_dir=transform_training.outputs['output'],
      train_steps=10000,
      eval_steps=5000,
      warm_starting=True
      )
```

Trainer invokes a training module, which is specified in the `module_file`
parameter.  A typical training module looks like this:

```python
# TFX will call this function
def trainer_fn(hparams, schema):
  """Build the estimator using the high level API.

  Args:
    hparams: Holds hyperparameters used to train the model as name/value pairs.
    schema: Holds the schema of the training examples.

  Returns:
    The estimator that will be used for training and eval
  """
  # Number of nodes in the first layer of the DNN
  first_dnn_layer_size = 100
  num_dnn_layers = 4
  dnn_decay_factor = 0.7

  train_batch_size = 40
  eval_batch_size = 40

  train_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      hparams.train_files,
      hparams.tf_transform_dir,
      batch_size=train_batch_size)

  eval_input_fn = lambda: _input_fn(  # pylint: disable=g-long-lambda
      hparams.eval_files,
      hparams.tf_transform_dir,
      batch_size=eval_batch_size)

  train_spec = tf.estimator.TrainSpec(  # pylint: disable=g-long-lambda
      train_input_fn,
      max_steps=hparams.train_steps)

  serving_receiver_fn = lambda: _example_serving_receiver_fn(  # pylint: disable=g-long-lambda
      hparams.tf_transform_dir, schema)

  exporter = tf.estimator.FinalExporter('chicago-taxi', serving_receiver_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=hparams.eval_steps,
      exporters=[exporter],
      name='chicago-taxi-eval')

  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=999, keep_checkpoint_max=1)

  run_config = run_config.replace(model_dir=hparams.serving_model_dir)

  estimator = _build_estimator(
      tf_transform_dir=hparams.tf_transform_dir,

      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
          for i in range(num_dnn_layers)
      ],
      config=run_config,
      warm_start_from=hparams.warm_start_from)

  # Input receiver for TFMA
  receiver_fn = lambda: _eval_input_receiver_fn(  # pylint: disable=g-long-lambda
      hparams.tf_transform_dir, schema)

  return TrainingSpec(estimator, train_spec, eval_spec, receiver_fn)


def _build_estimator(tf_transform_dir,
                     config,
                     hidden_units=None,
                     warm_start_from=None):
  """Build an estimator for predicting the tipping behavior of taxi riders.

  Args:
    tf_transform_dir: directory in which the tf-transform model was written
      during the preprocessing step.
    config: tf.contrib.learn.RunConfig defining the runtime environment for the
      estimator (including model_dir).
    hidden_units: [int], the layer sizes of the DNN (input layer first)
    warm_start_from: Optional directory to warm start from.

  Returns:
    Resulting DNNLinearCombinedClassifier.
  """
  metadata_dir = os.path.join(tf_transform_dir,
                              transform_fn_io.TRANSFORMED_METADATA_DIR)
  transformed_metadata = metadata_io.read_metadata(metadata_dir)
  transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

  transformed_feature_spec.pop(_transformed_name(_LABEL_KEY))

  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
  ]
  categorical_columns = [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
      for key in _transformed_names(_VOCAB_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
      for key in _transformed_names(_BUCKET_FEATURE_KEYS)
  ]
  categorical_columns += [
      tf.feature_column.categorical_column_with_identity(
          key, num_buckets=num_buckets, default_value=0)
      for key, num_buckets in zip(
          _transformed_names(_CATEGORICAL_FEATURE_KEYS),  #
          _MAX_CATEGORICAL_FEATURE_VALUES)
  ]
  return tf.estimator.DNNLinearCombinedClassifier(
      config=config,
      linear_feature_columns=categorical_columns,
      dnn_feature_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25],
      warm_start_from=warm_start_from)
```
