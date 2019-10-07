# The StatisticsGen TFX Pipeline Component

The StatisticsGen TFX pipeline component generates features statistics
over both training and serving data, which can be used by other pipeline
components.
StatisticsGen uses Beam to scale to large datasets.

* Consumes: datasets created by an ExampleGen pipeline component.
* Emits: Dataset statistics.

## StatisticsGen and TensorFlow Data Validation

StatisticsGen makes extensive use of [TensorFlow Data Validation](tfdv.md) for
generating statistics from your dataset.

## Using the StatsGen Component

A StatisticsGen pipeline component is typically very easy to deploy and
requires little
customization. Typical code looks like this:

```python
from tfx import components

...

compute_eval_stats = components.StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats'
      )
```
