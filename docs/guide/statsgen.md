# The StatisticsGen TFX Pipeline Component

The StatisticsGen component generates feature statistics
over `tf.Record` datasets. This statistical data can be used by other pipeline
components. While it is largely assumed that StatisticsGen will be
used with ExampleGen, it is possible for StatisticsGen to be used with
other components as well (As long as the data provided is a
`tf.Record` dataset). For large datasets, StatisticsGen uses Beam.

* Consumes: `tf.Record` datasets
* Emits: Dataset statistics

## StatisticsGen and TensorFlow Data Validation

StatisticsGen makes extensive use of the [TensorFlow Data Validation](tfdv.md) library for
generating statistics from a dataset

## Using the StatisticsGen Component

A StatisticsGen component is typically very easy to deploy and
requires little
customization. Typical code looks like this:

```python
from tfx import components

...

compute_example_statistics = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      instance_name='compute_example_statistics'
      )
```
