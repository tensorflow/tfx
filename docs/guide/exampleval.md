# The ExampleValidator TFX Pipeline Component

The ExampleValidator pipeline component identifies anomalies in training and serving
data. It can detect different classes of anomalies in the data. For example it
can:

1.  perform validity checks by comparing data statistics against a schema that
    codifies expectations of the user
1.  detect training-serving skew by comparing training and serving
    data.
1.  detect data drift by looking at a series of data.

The ExampleValidator pipeline component identifies any anomalies in the example data
by comparing data statistics computed by the StatisticsGen pipeline component against a
schema. The inferred schema codifies properties which the input data is expected to
satisfy, and can be modified by the developer.

* Consumes: A schema from a SchemaGen component, and statistics from a StatisticsGen
component.
* Emits: Validation results

## ExampleValidator and TensorFlow Data Validation

ExampleValidator makes extensive use of [TensorFlow Data Validation](tfdv.md)
for validating your input data.

## Using the ExampleValidator Component

An ExampleValidator pipeline component is typically very easy to deploy and
requires little customization. Typical code looks like this:

```python
validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema']
      )
```

More details are available in the
[ExampleValidator API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ExampleValidator).
