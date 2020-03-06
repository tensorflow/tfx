# The Pusher TFX Pipeline Component

The Pusher component is used to push a validated model to a
[deployment target](index.md#deployment_targets) during model training or
re-training.
It relies on a [Evaluator](evaluator.md) component to ensure that the new
model is "good enough" to be pushed to production.

* Consumes: A Trained model in [SavedModel](
https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model) format
* Emits: The same SavedModel, along with versioning metadata

## Using the Pusher Component

A Pusher pipeline component is typically very easy to deploy and requires little
customization, since all of the work is done by the Pusher TFX component.
Typical code looks like this:

```python
from tfx import components

...

pusher = components.Pusher(
  model=trainer.outputs['model'],
  model_blessing=model_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```
