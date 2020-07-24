# Designing TensorFlow Modeling Code For TFX

When designing your TensorFlow modeling code for TFX there are a few items to be
aware of, including the choice of a modeling API.

* Consumes: SavedModel from [Transform](transform.md), and data from
[ExampleGen](examplegen.md)
* Emits: Trained model in SavedModel format

<aside class="note" id="tf2-support"><b>Note:</b> TFX supports nearly all of
  TensorFlow 2.X, with minor exceptions. TFX also fully supports TensorFlow
  1.15.

<ul>
  <li>New TFX pipelines should use TensorFlow 2.x with Keras models via the
    <a href="https://github.com/tensorflow/community/blob/master/rfcs/20200117-tfx-generic-trainer.md">Generic Trainer</a>.</li>
  <li>Full support for TensorFlow 2.X, including improved support for
    tf.distribute, will be added incrementally in upcoming releases.</li>
  <li>Previous TFX pipelines can continue to use TensorFlow 1.15. To switch them
    to TensorFlow 2.X, see the
    <a href="https://www.tensorflow.org/guide/migrate">TensorFlow migration guide</a>.</li>
</ul>

To keep up to date on TFX releases, see the
<a href="https://github.com/tensorflow/tfx/blob/master/ROADMAP.md">TFX OSS
Roadmap</a>, read
<a href="https://blog.tensorflow.org/search?label=TFX&max-results=20">the TFX
blog</a> and subscribe to the
<a href="https://services.google.com/fb/forms/tensorflow/">TensorFlow
newsletter</a>.

</aside>

Your model's input layer should consume from the SavedModel that was created by
a [Transform](transform.md) component, and the layers of the Transform model should
be included with your model so that when you export your SavedModel and
EvalSavedModel they will include the transformations that were created by the
[Transform](transform.md) component.

A typical TensorFlow model design for TFX looks like this:

```python
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
