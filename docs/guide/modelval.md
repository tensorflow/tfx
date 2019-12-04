# The ModelValidator TFX Pipeline Component

The ModelValidator TFX Pipeline Component helps you validate your exported
models, ensuring that they are "good enough" to be pushed to production.

ModelValidator compares new models against a baseline (such as the currently
serving model) to determine if they're "good enough" relative to the baseline.
It does so by evaluating both models on an eval dataset and computing their
performance on metrics (e.g. AUC, loss). If the new model's metrics meet
developer-specified criteria relative to the baseline model (e.g. AUC is not lower),
the model is "blessed" (marked as good), indicating to the [Pusher](pusher.md)
that it is ok to push the model to production.

Note: Currently developers can only specify criteria metrics for the whole
evaluation split (dataset).  A future version will support more granular
criteria such as slices.

*   Consumes:
  * An eval split from ExampleGen
  * A trained model from Trainer
*   Emits: Validation results to [ML Metadata](mlmd.md)

## Using the ModelValidator Component

Typical code looks like this:

```python
import tfx
import tensorflow_model_analysis as tfma
from tfx.components.model_validator.component import ModelValidator

...

model_validator = ModelValidator(
      examples=example_gen.outputs['output_data'],
      model=trainer.outputs['model'])
```
