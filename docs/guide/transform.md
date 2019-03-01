# The Transform TFX Pipeline Component

The Transform TFX pipeline component performs feature engineering on tf.Examples
emitted from an [ExampleGen](examplegen.md) component, using a data schema created
by a [SchemaGen](schemagen.md) component, and emits a SavedModel.  When executed,
the SavedModel will accept tf.Examples emitted from an ExampleGen component and emit
the transformed feature data.

* Consumes: tf.Examples from an ExampleGen component, and a data schema from a
SchemaGen component.
* Emits: A SavedModel to a Trainer component

## Transform and TensorFlow Transform

Transform makes extensive use of [TensorFlow Transform](tft.md) for performing
feature engineering on your dataset.

## Developing a Transform Component

A Transform pipeline component is typically very easy to develop and requires little
customization, since all of the work is done by the Transform TFX component.
Note that a typical TFX pipeline will include two Transform components, one each for
training and evaluation datasets.  Transform code might look like this:

```python
from tfx import components

...

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[_transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in _VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[_transformed_name(
        key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

  for key in _BUCKET_FEATURE_KEYS:
    outputs[_transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

  for key in _CATEGORICAL_FEATURE_KEYS:
    outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
  tips = _fill_in_missing(inputs[_LABEL_KEY])
  outputs[_transformed_name(_LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64))

  return outputs

transform_training = components.Transform(
    input_data=examples_gen.outputs.training_examples,
    schema=infer_schema.outputs.output,
    module_file=taxi_pipeline_utils,
    name='transform-training')

transform_eval = components.Transform(
    input_data=examples_gen.outputs.eval_examples,
    schema=infer_schema.outputs.output,
    transform_dir=transform_training.outputs.output,
    name='transform-eval')
```
