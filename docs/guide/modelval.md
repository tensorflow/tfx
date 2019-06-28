# The ModelValidator TFX Pipeline Component

The ModelValidator TFX Pipeline Component helps you validate your exported models,
ensuring that they are "good enough" to be pushed to production.

ModelValidator compares new models against a baseline (such as the currently serving
model) to determine if they're "good enough" relative to the baseline. It does so by
evaluating both models on a dataset (e.g. holdout data, or a golden data set) and computing
their performance on metrics (e.g. AUC, loss). If the new model's metrics meet user-specified
criteria relative to the baseline model (e.g. AUC is not lower), the model is "blessed"
(marked as good), indicating to the [Pusher](pusher.md) that it is ok to push the model
to production.

*   Consumes: A schema from a SchemaGen component, and statistics from a
    StatisticsGen component.
*   Emits: Validation results to [TensorFlow Metadata](mlmd.md)

## Using the ModelValidator Component

An ModelValidator pipeline component is typically very easy to deploy and
requires little customization, since all of the work is done by the
ModelValidator TFX component. Typical code looks like this:

```python
from tfx import components
import tensorflow_model_analysis as tfma

...

# For model validation
taxi_mv_spec = [tfma.SingleSliceSpec()]

model_validator = components.ModelValidator(
      examples=examples_gen.outputs['output'],
      model=trainer.outputs['output'])
```
