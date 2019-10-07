# Custom TFX Component

Note: This guide is based on TFX 0.14.0 and requires TFX >= 0.14.0.

## Custom executor or custom component

If only custom processing logic is needed while the inputs, outputs, and
execution properties of the component are the same as an existing component, a
custom executor is sufficient. A custom component is needed when any of the
inputs, outputs, or execution properties are different than any existing TFX
components.

## How to create a custom component?

Developing a custom component will require:

*   A defined set of input and output artifact specifications for the new
    component. Specially, the types for the input artifacts should be consistent
    with the output artifact types of the components that produce the artifacts
    and the types for the output artifacts should be consistent with the input
    artifact types of the components that consume the artifacts if any.
*   The non-artifact execution parameters that are needed for the new component.

### ComponentSpec

The `ComponentSpec` class defines the component contract by defining the input
and output artifacts to a component as well as the parameters that will be used
for the component execution. There are three parts in it:

*   *INPUTS*: Specifications for the input artifacts that will be passed into
    the component executor. Input artifacts are often outputs from upstream
    components and thus share the same spec
*   *OUTPUTS*: Specifications for the output artifacts which the component will
    produce.
*   *PARAMETERS*: Specifications for the execution properties that will be
    passed into the component executor. These are non-artifact parameters that
    should be defined flexibly in the pipeline DSL and passed to the new
    component instance.

Here is an example of the ComponentSpec, the full example can be found in the
[TFX GitHub repo](https://github.com/tensorflow/tfx/blob/r0.14/tfx/examples/custom_components/slack/slack_component/executor.py).

```python
class SlackComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Slack Component."""

  INPUTS = {
      'model_export': ChannelParameter(type=standard_artifacts.Model),
      'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  OUTPUTS = {
      'slack_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing),
  }
  PARAMETERS = {
      'slack_token': ExecutionParameter(type=Text),
      'slack_channel_id': ExecutionParameter(type=Text),
      'timeout_sec': ExecutionParameter(type=int),
  }
```

### Executor

Next, write the executor code for the new component. Basically, a new subclass
of `base_executor.BaseExecutor` needs to be created with its `Do` function
overriden. In the `Do` function, the arguments `input_dict`, `output_dict` and
`exec_properties` that are passed in map to `INPUTS`, `OUTPUTS` and `PARAMETERS`
that are defined in ComponentSpec respectively. For `exec_properties`, the value
can be fetched directly through a dictionary lookup. For artifacts in
`input_dict` and `output_dict`, there are convenient functions available to
fetch the URIs of the artifacts (see `model_export_uri` and `model_blessing_uri`
in the example) or get the artifact object (see `slack_blessing` in the
example).

```python

class Executor(base_executor.BaseExecutor):
  """Executor for Slack component."""
  ...
  def Do(self, input_dict: Dict[Text, List[types.TfxArtifact]],
         output_dict: Dict[Text, List[types.TfxArtifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...
    # Fetch execution properties from exec_properties dict.
    slack_token = exec_properties['slack_token']
    slack_channel_id = exec_properties['slack_channel_id']
    timeout_sec = exec_properties['timeout_sec']

    # Fetch input URIs from input_dict.
    model_export_uri = types.get_single_uri(input_dict['model_export'])
    model_blessing_uri = types.get_single_uri(input_dict['model_blessing'])

    # Fetch output artifact from output_dict.
    slack_blessing =
        types.get_single_instance(output_dict['slack_blessing'])
    ...
```

The example above only shows the part of the implementation that uses the
passed-in value. Please see the full example in the
[TFX GitHub repo](https://github.com/tensorflow/tfx/blob/r0.14/tfx/examples/custom_components/slack/slack_component/executor.py).

#### Unit testing a custom executor

Unit tests for the custom executor can be created similar to
[this one](https://github.com/tensorflow/tfx/blob/r0.14/tfx/components/transform/executor_test.py).

### Component interface

Now that the most complex part is complete, the next step is to assemble these
pieces into a component interface, to enable the component to be used in a
pipeline. There are several steps:

*   Make the component interface a subclass of `base_component.BaseComponent`
*   Assign a class variable `SPEC_CLASS` with the `ComponentSpec` class that was
    defined earlier
*   Assign a class variable `EXECUTOR_SPEC` with the Executor class that was
    defined earlier
*   Define the `__init__()` constructor function by using the arguments to the
    function to construct an instance of the ComponentSpec class and invoke the
    super function with that value, along with an optional name

When an instance of the component is created, type checking logic in the
`base_component.BaseComponent` class will be invoked to ensure that the
arguments which were passed in are compatible with the type info defined in the
`ComponentSpec` class.

```python
from slack_component import executor

class SlackComponent(base_component.BaseComponent):
  """Custom TFX Slack Component."""

  SPEC_CLASS = SlackComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               model_export: channel.Channel,
               model_blessing: channel.Channel,
               slack_token: Text,
               slack_channel_id: Text,
               timeout_sec: int,
               slack_blessing: Optional[channel.Channel] = None,
               name: Optional[Text] = None):
    slack_blessing = slack_blessing or channel.Channel(
        type_name='ModelBlessingPath',
        artifacts=[types.TfxArtifact('ModelBlessingPath')])
    spec = SlackComponentSpec(
        slack_token=slack_token,
        slack_channel_id=slack_channel_id,
        timeout_sec=timeout_sec,
        model=model_export,
        model_blessing=model_blessing,
        slack_blessing=slack_blessing)
    super(SlackComponent, self).__init__(spec=spec, name=name)
```

### Assemble into a TFX pipeline

The last step is to plug the new custom component into a TFX pipeline. Besides
adding an instance of the new component, the following are also needed:

*   Properly wire the upstream and downstream components of the new component to
    it. This is done by referencing the outputs of the upstream component in the
    new component and referencing the outputs of the new component in downstream
    components
*   Add the new component instance to the components list when constructing the
    pipeline.

The example below highlights the aforementioned changes. Full example can be
found in the
[TFX GitHub repo](https://github.com/tensorflow/tfx/blob/r0.14/tfx/examples/custom_components/slack/slack_component/executor.py).

```python
def _create_pipeline():
  ...
  model_validator = ModelValidator(
      examples=example_gen.outputs['examples'], model=trainer.outputs['model'])

  slack_validator = SlackComponent(
      model=trainer.outputs['model'],
      model_blessing=model_validator.outputs['blessing'],
      slack_token=_slack_token,
      slack_channel_id=_slack_channel_id,
      timeout_sec=3600,
  )

  pusher = Pusher(
      ...
      model_blessing=slack_validator.outputs['slack_blessing'],
      ...)

  return pipeline.Pipeline(
      ...
      components=[
          ..., model_validator, slack_validator, pusher
      ],
      ...
  )
```

## Deploy a custom component

Beside code changes, all the newly added parts (`ComponentSpec`, `Executor`,
component interface) need to be accessible in pipeline running environment in
order to run the pipeline properly.
