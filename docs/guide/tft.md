# Preprocessing and Transforming Your Data

## Introduction

TensorFlow Transform is a great tool for transforming feature data before it goes to
your model and as a part of the training process. Common feature transformations
include:

* **Embedding**: converting sparse features (like the integer IDs produced by a
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

For more information about TensorFlow Transform, especially as a stand alone
outside of TFX, see the [getting started
docs](https://github.com/tensorflow/transform/blob/master/docs/get_started.md).

## The TensorFlow Transform `preprocessing_fn`

In `preprocessing_fn` you define a series of functions that manipulate the input
dict of tensors to produce the output dict of tensors. You can find helper
functions like scale_to_0_1 and compute_and_apply_vocabulary the [TensorFlow
Transform
API](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv) or use
regular TensorFlow functions as shown below.

```python
def preprocessing_fn(inputs):
  """Preprocess input features into transformed features."""
  outputs = {}

  outputs['my_feature'] = tf.some_complex_tensorflow_ops(inputs['my_feature'])

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

In the Transform component, both reading data from disk and parsing using the
above mentioned feature spec is done automatically and the user only needs to
define the `preprocessing_fn`. In your TensorFlow modeling code you then need to
call a TensorFlow Transform helper to get the feature spec and then use this to
do reading/parsing of data in your `input_fn` (see [Training on TensorFlow
Transform Output](#training-on-tftransform-output)).

The exact rules for inferring a feature spec from the `Schema` are shown below.
The table shows how the fields of the `Feature` proto affect the corresponding
value of the feature spec for that feature. You can look up each feature in the
table below, to see what kind of parsing spec will be used for that feature.

WARNING: If the schema changes you need to rerun Transform to regenerate the
metadata that's used during parsing.

value_count                 | presence.min_fraction | type  | feature spec
--------------------------- | --------------------- | ----- | ------------
min = max = 1               | 1                     | BYTES | `FixedLenFeature(shape=(), dtype=tf.string, default_value=None)`
min = max = 1               | 1                     | INT   | `FixedLenFeature(shape=(), dtype=tf.int64, default_value=None)`
min = max = 1               | 1                     | FLOAT | `FixedLenFeature(shape=(), dtype=tf.float32, default_value=None)`
min = max = 1               | <1                    | BYTES | `FixedLenFeature(shape=(), dtype=tf.string, default_value='')`
min = max = 1               | <1                    | INT   | `FixedLenFeature(shape=(), dtype=tf.int64, default_value=-1)`
min = max = 1               | <1                    | FLOAT | `FixedLenFeature(shape=(), dtype=tf.float32, default_value=-1)`
min = max = k > 1           | 1                     | BYTES | `FixedLenFeature(shape=(k,), dtype=tf.string, default_value=None)`
min = max = k > 1           | 1                     | INT   | `FixedLenFeature(shape=(k,), dtype=tf.int64, default_value=None)`
min = max = k > 1           | 1                     | FLOAT | `FixedLenFeature(shape=(k,), dtype=tf.float32, default_value=None)`
min = max = k > 1           | <1                    | BYTES | `FixedLenFeature(shape=(k,), dtype=tf.string, default_value='')`
min = max = k > 1           | <1                    | INT   | `FixedLenFeature(shape=(k,), dtype=tf.int64, default_value=-1)`
min = max = k > 1           | <1                    | FLOAT | `FixedLenFeature(shape=(k,), dtype=tf.float32, default_value=-1)`
min = max = 0 or min != max | any                   | BYTES | `VarLenFeature(dtype=tf.string)`
min = max = 0 or min != max | any                   | INT   | `VarLenFeature(dtype=tf.int64)`
min = max = 0 or min != max | any                   | FLOAT | `VarLenFeature(dtype=tf.float32)`

These rules can be summarized as follows

-   If `value_count.min = value_count.max = 1` then construct a
    `FixedLenFeature` with shape `()`, i.e. a scalar-valued feature.
-   If `value_count.min = value_count.max = k > 1` then construct a
    `FixedLenFeature` with shape `(k,)`, i.e. a vector-valued feature.
-   For other cases, construct a `VarLenFeature`.
-   The `dtype` is based on the `type` field, with `BYTES`, `INT` and `FLOAT`
    mapping to `tf.string`, `tf.int64` and `tf.float32` respectively.
-   If constructing `FixedLenFeature` we set `default_value` if
    `presence.min_fraction < 1`. In this case, the default_value is determined
    by the feature's type, with `''` for `BYTES` and `-1` for `INT` and `FLOAT.

WARNING: When using a schema that results in a `VarLenFeature`, the
`dense_shape` of the resulting `SparseTensor` will vary per-batch, and will have
no relation to `value_count.max`. If you convert this to a `Tensor`, you should
ensure the resulting `Tensor` has a constant shape (except for the
batch-dimension).

{#workarounds-for-overriding-the-default-value}
### Workarounds for overriding the default value 

There is no way to change the default value in the feature spec that Transform
infers. However there are the following workarounds.

The first workaround is to make use of the given default value in the code, for
example using `tf.where` to handle the case where the feature equals the default
value. We advise against using `tf.cond` as control flow ops have complex
interactions with TensorFlow Transform. Note that there is no way to distinguish
a
missing value which was set to the default value, and a feature that was
originally set to the default value, as they result in identical tensors.

The second workaround is to change `value_count` in the `Schema`, e.g. change
from

```
  presence: <
    min_count: 1
  >
  value_count: <
    min: 1
    max: 1
  >
  type: BYTES
```

which produces a `FixedLenFeature(shape=(), dtype=tf.string, default_value='')`,
to

```
  presence: <
    min_count: 1
  >
  value_count: <
    min: 0
    max: 1
  >
  type: BYTES
```

which produces a `VarLenFeature(dtype=tf.string)`. This will then result in
`SparseTensor` of shape `(None, None)` where the size in the first dimension is
the batch size and the size in the second dimension is the smallest size that
will fit the values of the batch.

Note that since the second dimension of the `dense_shape` of the `SparseTensor`
will be the smallest size that fits the values of the batch, in the case above
the `dense_shape` will sometimes be `(batch_size, 1)` and sometimes
`(batch_size, 0)`. The case of `(batch_size, 0)` occurs when the feature is
missing in every example in the batch. Because of this, you should be careful
about using `tf.sparse_tensor_to_dense` as you will end up with a `Tensor` whose
shape varies across batches. Instead, you should override the shape as follows

```python
def preprocessing_fn(inputs):
  ...
  # my_feature is a sparse_tensor
  my_feature = inputs['my_feature']
  # fixed_sparse_feature is a `Tensor` whose second dim is always 1.
  # sparse_to_dense will throw a runtime error if my_feature.dense_shape[1]
  # is greater than 1, which will occur if my_feature contains more than one
  # value for any example in the batch.
  my_feature_dense = tf.sparse_to_dense(
      sparse_indices=my_feature.indices,
      output_shape=[my_feature.dense_shape[0], 1],
      sparse_values=my_feature.values,
      default_value='MY_DEFAULT')
  ...
```

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
  _ = tft.uniques(education, vocab_filename=features.RAW_LABEL_KEY)

  ...
```

The preprocessing function above takes the raw input feature (which will also be
returned as part of the output of the preprocessing function) and calls
`tft.uniques` on it. This results in a vocabulary being generated for `education`
that can be accessed in the model.

The example also shows how to transform a label and then generate a vocabulary
for the transformed label. In particular it takes the raw label `education` and
converts all but the top 5 labels (by frequency) to `UNKNOWN`, without
converting the label to an integer.

In the model code, the classifier must be given the vocabulary generated by
`tft.uniques` as the `label_vocabulary` argument. This is done by first reading
this vocabulary as a list with a helper function. This is shown in the snippet
below. Note the example code uses the transformed label discussed above but
here we show code for using the raw label.

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
