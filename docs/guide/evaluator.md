# The Evaluator TFX Pipeline Component

The `Evaluator` component helps you understand how your model performs on different 
subsets of your data by performing deep analysis on the training
results of your models.

*   Consumes: `EvalSavedModel` from a [Trainer](trainer.md) component
*   Emits: Analysis results to [TensorFlow Metadata](mlmd.md)

## Evaluator and TensorFlow Model Analysis

`Evaluator` leverages the [TensorFlow Model Analysis](tfma.md) library to perform
model analysis. TFMA uses [Apache Beam](beam.md) for scalable processing.

## Using the Evaluator Component

An `Evaluator` component is typically very easy to deploy and requires little
customization.
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
      model_exports=trainer.outputs['model'],
      fairness_indicator_thresholds = [0.25, 0.5, 0.75])
```
