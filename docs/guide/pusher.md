# The Pusher TFX Pipeline Component

The Pusher component is used to push a validated model to a
[deployment target](index.md#deployment_targets) during model training or
re-training. Before the deployment, Pusher relies on one or more blessings from
other validation components to decide whether to push the model or not.

-   [Evaluator](evaluator) blesses the model if the new trained model is "good
    enough" to be pushed to production.
-   (Optional but recommended) [InfraValidator](infra_validator) blesses the
    model if the model is mechanically servable in a production environment.

A Pusher component consumes a trained model in [SavedModel](/guide/saved_model)
format, and produces the same SavedModel, along with versioning metadata.

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
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```
