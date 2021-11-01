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
pusher = Pusher(
  model=trainer.outputs['model'],
  model_blessing=evaluator.outputs['blessing'],
  infra_blessing=infra_validator.outputs['blessing'],
  push_destination=tfx.proto.PushDestination(
    filesystem=tfx.proto.PushDestination.Filesystem(
        base_directory=serving_model_dir)
  )
)
```

### Pushing a model produced from InfraValidator.

(From version 0.30.0)

InfraValidator can also produce `InfraBlessing` artifact containing a
[model with warmup](infra_validator#producing_a_savedmodel_with_warmup), and
Pusher can push it just like a `Model` artifact.

```python
infra_validator = InfraValidator(
    ...,
    # make_warmup=True will produce a model with warmup requests in its
    # 'blessing' output.
    request_spec=tfx.proto.RequestSpec(..., make_warmup=True)
)

pusher = Pusher(
    # Push model from 'infra_blessing' input.
    infra_blessing=infra_validator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(...)
)
```

More details are available in the
[Pusher API reference](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/Pusher).
