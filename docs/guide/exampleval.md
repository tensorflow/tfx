# The ExampleValidator TFX Pipeline Component

The ExampleValidator pipeline component identifies anomalies in training and serving
data. It can detect different classes of anomalies in the data. For example it
can:

1.  perform validity checks by comparing data statistics against a schema that
    codifies expectations of the user
1.  detect training-serving skew by comparing examples in training and serving
    data.
1.  detect data drift by looking at a series of data.

The ExampleValidator pipeline component identifies any anomalies in the example data
by comparing data statistics computed by the StatsGen pipeline component against a
schema. The inferred schema codifies properties which the input data is expected to
satisfy, and can be modified by the developer.

* Consumes: A schema from a SchemaGen component, and statistics from a StatsGen
component.
* Emits: Validation results to [TensorFlow Metadata](tfmd.md)

## ExampleValidator and TensorFlow Data Validation

ExampleValidator makes extensive use of [TensorFlow Data Validation](tfdv.md)
for validating your input data, which in turn use [Beam](beam.md) for scalable
processing.

## Developing an ExampleValidator Component

An ExampleValidator pipeline component is typically very easy to develop and
requires little customization, since all of the work is done by the
ExampleValidator TFX component. Typical code looks like this:

```python
from tfx import components

...

validate_stats = components.ExampleValidator(
      stats=compute_eval_stats.outputs.output,
      schema=infer_schema.outputs.output
      )
```
