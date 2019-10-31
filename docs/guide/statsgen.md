# The StatisticsGen TFX Pipeline Component

The `StatisticsGen` component generates feature statistics
over both training and serving data. This data can be used by other pipeline
components.
`StatisticsGen` uses Beam to scale to large datasets.

* Consumes: Datasets created by an `ExampleGen` component
* Emits: Dataset statistics

## StatisticsGen and TensorFlow Data Validation

`StatisticsGen` makes extensive use of the [TensorFlow Data Validation](tfdv.md) library while 
generating statistics from your dataset

## Using the StatisticsGen Component

A `StatisticsGen` component is typically very easy to deploy and
requires little
customization. Typical code looks like this:

```python
from tfx.components import StatisticsGen

...

compute_eval_stats = StatisticsGen(
      examples=example_gen.outputs['examples'],
      name='compute-eval-stats')
```
