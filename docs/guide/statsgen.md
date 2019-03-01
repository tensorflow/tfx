# The StatsGen TFX Pipeline Component

The StatsGen TFX pipeline component generates features statistics and random samples
over both training and serving data, which can be used by other pipeline components.
StatsGen uses Beam and approximate algorithms to scale to large datasets.

* Consumes: tf.Examples created by an ExampleGen pipeline component, or CSV directly.
* Emits: Dataset statistics in metadata

## StatsGen and TensorFlow Data Validation

StatsGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for
generating statistics from your dataset.

## Developing a StatsGen Component

A StatsGen pipeline component is typically very easy to develop and requires little
customization, since all of the work is done by the StatsGen TFX component.
Typical code looks like this:

```python
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      input_data=examples_gen.outputs.eval_examples,
      name='compute-eval-stats'
      )
```
