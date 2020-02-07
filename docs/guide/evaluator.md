# The Evaluator TFX Pipeline Component

The Evaluator TFX pipeline component performs deep analysis on the training
results for your models, to help you understand how your model performs on
subsets of your data.

*   Consumes: EvalSavedModel from [Trainer](trainer.md)
*   Emits: Analysis results to [ML Metadata](mlmd.md)

## Evaluator and TensorFlow Model Analysis

Evaluator leverages the [TensorFlow Model Analysis](tfma.md) library to perform
the analysis, which in turn use [Apache Beam](beam.md) for scalable processing.

## Using the Evaluator Component

A Evaluator pipeline component is typically very easy to deploy and requires little
customization, since all of the work is done by the Evaluator TFX component.
Typical code looks like this:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

# For TFMA evaluation
taxi_eval_spec = [
    tfma.SingleSliceSpec(),
    tfma.SingleSliceSpec(columns=['trip_start_hour'])
]

model_analyzer = components.Evaluator(
      examples=examples_gen.outputs['examples'],
      feature_slicing_spec=taxi_eval_spec,
      model=trainer.outputs['model'],
      fairness_indicator_thresholds = [0.25, 0.5, 0.75]
      )
```
