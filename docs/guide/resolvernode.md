# The ResolverNode

A `ResolverNode` is a special TFX node. Unlike other component nodes, it does
not process or generate any new data or model. Instead, it takes `Channel`s as
the search scope and uses specified resolution logic defined by a `Resolver`
class to get the desired artifacts so that its downstream pipeline nodes can use
those artifacts as the input artifacts.

A `ResolverNode` takes the following arguments:

*   `instance_name`: The name of the `ResolverNode` instance.
*   `resolver_class`: A `BaseResolver` subclass which contains the artifact
    resolution logic.
*   `resolver_configs`: A dict of key to Jsonable type representing configs that
    will be used to construct the `Resolver`.
*   `**kwargs`: A key -> `Channel` dict, describing what are the `Channel`s to
    be resolved. This is set by user through keyword args.

## Resolver

A `Resolver` class should inherit from `BaseResolver` class which is defined as
the following:

```Python
class BaseResolver(with_metaclass(abc.ABCMeta, object)):

  @abc.abstractmethod
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> ResolveResult:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      pipeline_info: PipelineInfo of the current pipeline. We do not want to
        query artifacts across pipeline boundary.
      metadata_handler: a read-only handler to query MLMD.
      source_channels: a key -> channel dict which contains the info of the
        source channels.

    Returns:
      a ResolveResult instance.

    """
    raise NotImplementedError
```

## Using the ResolverNode

To define an `ResolverNode`, typical code looks like the following:

```Python
from tfx.components import ResolverNode

model_resolver = ResolverNode(
    instance_name='latest_blessed_model_resolver',
    resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
    model=Channel(type=Model, producer_component_id=Trainer.get_id()),
    model_blessing=Channel(
        type=ModelBlessing, producer_component_id=Evaluator.get_id()))
```

Using the output of an `ResolverNode` is similar to using the output of any
component node. The only difference is that the output keys are not static, but
are the same as the keys to the `Channel`s provided to the `ResolverNode`. For
example, in the above code snippet, there are two `Channel`s provided to the
`model_resolver`:

*   `model` is the key of the `Channel`s for the `Model` artifacts search scope
*   `model_blessing` is the key of the `Channel` for the `ModelBlessing`
    artifacts search scope.

When downstream nodes use the output of `model_resolver`, they also use the same
key to refer to the resolution result in the same search scope. For example, the
code snippet below shows a `Evaluator` uses the output of `model_resolver` with
the `model` key to get the latest blessed `Model` artifact:

```Python
model_analyzer = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
```
