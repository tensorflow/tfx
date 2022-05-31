# The Transform TFX Pipeline Component

The Transform TFX pipeline component performs feature engineering on tf.Examples
emitted from an [ExampleGen](examplegen.md) component, using a data schema created
by a [SchemaGen](schemagen.md) component, and emits both a SavedModel as well as
statistics on both pre-transform and post-transform data.  When executed,
the SavedModel will accept tf.Examples emitted from an ExampleGen component and emit
the transformed feature data.

* Consumes: tf.Examples from an ExampleGen component, and a data schema from a
SchemaGen component.
* Emits: A SavedModel to a Trainer component, pre-transform and post-transform
statistics.

## Configuring a Transform Component

Once your `preprocessing_fn` is written, it needs to be defined in a python
module that is then provided to the Transform component as an input.  This
module will be loaded by transform and the function named `preprocessing_fn`
will be found and used by Transform to construct the preprocessing pipeline.

```
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_taxi_transform_module_file))
```

Additionally, you may wish to provide options to the [TFDV](tfdv.md)-based
pre-transform or post-transform statistics computation. To do so, define a
`stats_options_updater_fn` within the same module.

## Transform and TensorFlow Transform

Transform makes extensive use of [TensorFlow Transform](tft.md) for performing
feature engineering on your dataset.  TensorFlow Transform is a great tool for
transforming feature data before it goes to your model and as a part of the
training process. Common feature transformations include:

*   **Embedding**: converting sparse features (like the integer IDs produced by a
    vocabulary) into dense features by finding a meaningful mapping from high-
    dimensional space to low dimensional space. See the [Embeddings unit in the
    Machine-learning Crash Course](
    https://developers.google.com/machine-learning/crash-course/embedding)
    for an introduction to embeddings.
*   **Vocabulary generation**: converting strings or other non-numeric features
    into integers by creating a vocabulary that maps each unique value to an ID
    number.
*   **Normalizing values**: transforming numeric features so that they all fall
    within a similar range.
*   **Bucketization**: converting continuous-valued features into categorical
    features by assigning values to discrete buckets.
*   **Enriching text features**: producing features from raw data like tokens,
    n-grams, entities, sentiment, etc., to enrich the feature set.

TensorFlow Transform provides support for these and many other kinds of
transformations:

* Automatically generate a vocabulary from your latest data.

* Perform arbitrary transformations on your data before sending it to your
  model. TensorFlow Transform builds transformations into the TensorFlow graph for
  your model so the same transformations are performed at training and inference
  time. You can define transformations that refer to global properties of the
  data, like the max value of a feature across all training instances.

You can transform your data however you like prior to running TFX.
But if you do it within TensorFlow Transform, transforms
become part of the TensorFlow graph. This approach helps avoid training/serving
skew.

Transformations inside your modeling code use FeatureColumns.
Using FeatureColumns, you can define bucketizations, integerizations that use
predefined vocabularies, or any other transformations that can be defined
without looking at the data.

By contrast, TensorFlow Transform is designed for transformations that require a
full pass over the data to compute values that are not known in advance. For
example, vocabulary generation requires a full pass over the data.

Note: These computations are implemented in [Apache Beam](https://beam.apache.org/)
under the hood.

In addition to computing values using Apache Beam, TensorFlow Transform allows
users to embed these values into a TensorFlow graph, which can then be loaded
into the training graph. For example when normalizing features, the
`tft.scale_to_z_score` function will compute the mean and standard deviation of
a feature, and also a representation, in a TensorFlow graph, of the function
that subtracts the mean and divides by the standard deviation. By emitting a
TensorFlow graph, not just statistics, TensorFlow Transform simplifies the
process of authoring your preprocessing pipeline.

Since the preprocessing is expressed as a graph, it can happen on the server,
and it's guaranteed to be consistent between training and serving. This
consistency eliminates one source of training/serving skew.

TensorFlow Transform allows users to specify their preprocessing pipeline using
TensorFlow code. This means that a pipeline is constructed in the same manner as
a TensorFlow graph. If only TensorFlow ops were used in this graph, the pipeline
would be a pure map that accepts batches of input and returns batches of output.
Such a pipeline would be equivalent to placing this graph inside your `input_fn`
when using the `tf.Estimator` API. In order to specify full-pass operations such
as computing quantiles, TensorFlow Transform provides special functions called
`analyzers` that appear like TensorFlow ops, but in fact specify a deferred
computation that will be done by Apache Beam, and the output inserted into the
graph as a constant. While an ordinary TensorFlow op will take a single batch as
its input, perform some computation on just that batch and emit a batch, an
`analyzer` will perform a global reduction (implemented in Apache Beam) over all
batches and return the result.

By combining ordinary TensorFlow ops and TensorFlow Transform analyzers, users
can create complex pipelines to preprocess their data. For example the
`tft.scale_to_z_score` function takes an input tensor and returns that tensor
normalized to have mean `0` and variance `1`. It does this by calling the `mean`
and `var` analyzers under the hood, which will effectively generate constants in
the graph equal to the mean and variance of the input tensor. It will then use
TensorFlow ops to subtract the mean and divide by the standard deviation.

## The TensorFlow Transform `preprocessing_fn`

The TFX Transform component simplifies the use of Transform by handling the API
calls related to reading and writing data, and writing the output SavedModel to
disk.  As a TFX user, you only have to define a single function called the
`preprocessing_fn`.
In `preprocessing_fn` you define a series of functions that manipulate the input
dict of tensors to produce the output dict of tensors. You can find helper
functions like scale_to_0_1 and compute_and_apply_vocabulary the
[TensorFlow Transform API](/tfx/transform/api_docs/python/tft) or use
regular TensorFlow functions as shown below.

```python
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in _DENSE_FLOAT_FEATURE_KEYS:
    # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
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
```

### Understanding the inputs to the preprocessing_fn

The `preprocessing_fn` describes a series of operations on tensors (that is,
`Tensor`s or `SparseTensor`s) and so to write the `preprocessing_fn` correctly
it is necessary to understand how your data is represented as tensors. The input
to the `preprocessing_fn` is determined by the schema. A `Schema` proto contains
a list of `Feature`s, and Transform converts these to a "feature spec"
(sometimes called a "parsing spec") which is a dict whose keys are feature names
and whose values are one of `FixedLenFeature` or `VarLenFeature` (or other
options not used by TensorFlow Transform).

The rules for inferring a feature spec from the `Schema` are

-   Each `feature` with `shape` set will result in a `tf.FixedLenFeature` with
    shape and `default_value=None`.  `presence.min_fraction` must be `1`
    otherwise and error will result, since when there is no default value, a
    `tf.FixedLenFeature` requires the feature to always be present.
-   Each `feature` with `shape` not set will result in a `VarLenFeature`.
-   Each `sparse_feature` will result in a `tf.SparseFeature` whose `size` and
    `is_sorted` are determined by the `fixed_shape` and `is_sorted` fields of
    the `SparseFeature` message.
-   Features used as the `index_feature` or `value_feature` of a
    `sparse_feature` will not have their own entry generated in the feature
    spec.
-   The correspondence between `type` field of the `feature` (or the values
    feature of a `sparse_feature` proto) and the `dtype` of the feature spec is
    given by the following table:

`type`             | `dtype`
------------------ | ------------
`schema_pb2.INT`   | `tf.int64`
`schema_pb2.FLOAT` | `tf.float32`
`schema_pb2.BYTES` | `tf.string`

## Using TensorFlow Transform to handle string labels

Usually one wants to use TensorFlow Transform to both generate a vocabulary and
apply that vocabulary to convert strings to integers. When following this
workflow, the `input_fn` constructed in the model will output the integerized
string. However labels are an exception, because in order for the model to be
able to map the output (integer) labels back to strings, the model needs the
`input_fn` to output a string label, together with a list of possible values of
the label. E.g. if the labels are `cat` and `dog` then the output of the
`input_fn` should be these raw strings, and the keys `["cat", "dog"]` need to be
passed into the estimator as a parameter (see details below).

In order to handle the mapping of string labels to integers, you should use
TensorFlow Transform to generate a vocabulary. We demonstrate this in the code
snippet below:

```python
def _preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""

  ...


  education = inputs[features.RAW_LABEL_KEY]
  _ = tft.vocabulary(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

The preprocessing function above takes the raw input feature (which will also be
returned as part of the output of the preprocessing function) and calls
`tft.vocabulary` on it. This results in a vocabulary being generated for
`education` that can be accessed in the model.

The example also shows how to transform a label and then generate a vocabulary
for the transformed label. In particular it takes the raw label `education` and
converts all but the top 5 labels (by frequency) to `UNKNOWN`, without
converting the label to an integer.

In the model code, the classifier must be given the vocabulary generated by
`tft.vocabulary` as the `label_vocabulary` argument. This is done by first
reading this vocabulary as a list with a helper function. This is shown in the
snippet below. Note the example code uses the transformed label discussed above
but here we show code for using the raw label.

```python
def create_estimator(pipeline_inputs, hparams):

  ...

  tf_transform_output = trainer_util.TFTransformOutput(
      pipeline_inputs.transform_dir)

  # vocabulary_by_name() returns a Python list.
  label_vocabulary = tf_transform_output.vocabulary_by_name(
      features.RAW_LABEL_KEY)

  return tf.contrib.learn.DNNLinearCombinedClassifier(
      ...
      n_classes=len(label_vocab),
      label_vocabulary=label_vocab,
      ...)
```

## Configuring pre-transform and post-transform statistics

As mentioned above, the Transform component invokes TFDV to compute both
pre-transform and post-transform statistics. TFDV takes as input an optional
[StatsOptions](https://github.com/tensorflow/data-validation/blob/master/tensorflow_data_validation/statistics/stats_options.py)
object. Users may wish to configure this object to enable certain additional
statistics (e.g. NLP statistics) or to set thresholds that are validated (e.g.
min / max token frequency). To do so, define a `stats_options_updater_fn` in the
module file.

```python
def stats_options_updater_fn(stats_type, stats_options):
  ...
  if stats_type == stats_options_util.StatsType.PRE_TRANSFORM:
    # Update stats_options to modify pre-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  if stats_type == stats_options_util.StatsType.POST_TRANSFORM
    # Update stats_options to modify post-transform statistics computation.
    # Most constraints are specified in the schema which can be accessed
    # via stats_options.schema.
  return stats_options
```

Post-transform statistics often benefit from knowledge of the vocabulary being
used for preprocessing a feature. The vocabulary name to path mapping is
provided to StatsOptions (and hence TFDV) for every TFT-generated vocabulary.
Additionally, mappings for externally-created vocabularies can be added by
either (i) directly modifying the `vocab_paths` dictionary within StatsOptions
or by (ii) using `tft.annotate_asset`.
