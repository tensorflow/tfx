# Building Fully Custom Components

This guide describes how to use the TFX API to build a fully custom component.
Fully custom components let you build components by defining the component
specification, executor, and component interface classes. This approach lets you
reuse and extend a standard component to fit your needs.

If you are new to TFX pipelines,
[learn more about the core concepts of TFX pipelines](understanding_tfx_pipelines).

## Custom executor or custom component

If only custom processing logic is needed while the inputs, outputs, and
execution properties of the component are the same as an existing component, a
custom executor is sufficient. A fully custom component is needed when any of
the inputs, outputs, or execution properties are different from any existing TFX
components.

## How to create a custom component?

Developing a fully custom component requires:

*   A defined set of input and output artifact specifications for the new
    component. Specially, the types for the input artifacts should be consistent
    with the output artifact types of the components that produce the artifacts
    and the types for the output artifacts should be consistent with the input
    artifact types of the components that consume the artifacts if any.
*   The non-artifact execution parameters that are needed for the new component.

### ComponentSpec

The `ComponentSpec` class defines the component contract by defining the input
and output artifacts to a component as well as the parameters that are used for
the component execution. It has three parts:

*   *INPUTS*: A dictionary of typed parameters for the input artifacts that are
    passed into the component executor. Normally input artifacts are the outputs
    from upstream components and thus share the same type.
*   *OUTPUTS*: A dictionary of typed parameters for the output artifacts which
    the component produces.
*   *PARAMETERS*: A dictionary of additional
    [ExecutionParameter](https://github.com/tensorflow/tfx/blob/54aa6fbec6bffafa8352fe51b11251b1e44a2bf1/tfx/types/component_spec.py#L274)
    items that will be passed into the component executor. These are
    non-artifact parameters that we want to define flexibly in the pipeline DSL
    and pass into execution.

Here is an example of the ComponentSpec:

```python
class HelloComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
      'name': ExecutionParameter(type=Text),
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'input_data': ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }
```

### Executor

Next, write the executor code for the new component. Basically, a new subclass
of `base_executor.BaseExecutor` needs to be created with its `Do` function
overriden. In the `Do` function, the arguments `input_dict`, `output_dict` and
`exec_properties` that are passed in map to `INPUTS`, `OUTPUTS` and `PARAMETERS`
that are defined in ComponentSpec respectively. For `exec_properties`, the value
can be fetched directly through a dictionary lookup. For artifacts in
`input_dict` and `output_dict`, there are convenient functions available in
[artifact_utils](https://github.com/tensorflow/tfx/blob/41823f91dbdcb93195225a538968a80ba4bb1f55/tfx/types/artifact_utils.py)
class that can be used to fetch artifact instance or artifact uri.

```python
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    ...

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = artifact_utils.get_split_uri([artifact], split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
```

#### Unit testing a custom executor

Unit tests for the custom executor can be created similar to
[this one](https://github.com/tensorflow/tfx/blob/r0.15/tfx/components/transform/executor_test.py).

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
from tfx.types import standard_artifacts
from hello_component import executor

class HelloComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component."""

  SPEC_CLASS = HelloComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               input_data: types.Channel = None,
               output_data: types.Channel = None,
               name: Optional[Text] = None):
    if not output_data:
      examples_artifact = standard_artifacts.Examples()
      examples_artifact.split_names = input_data.get()[0].split_names
      output_data = channel_utils.as_channel([examples_artifact])

    spec = HelloComponentSpec(input_data=input_data,
                              output_data=output_data, name=name)
    super(HelloComponent, self).__init__(spec=spec)
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
[TFX GitHub repo](https://github.com/tensorflow/tfx/tree/master/tfx/examples/custom_components/hello_world).

```python
def _create_pipeline():
  ...
  example_gen = CsvExampleGen(input_base=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], name='HelloWorld')
  statistics_gen = StatisticsGen(examples=hello.outputs['output_data'])
  ...
  return pipeline.Pipeline(
      ...
      components=[example_gen, hello, statistics_gen, ...],
      ...
  )
```

## Deploy a fully custom component

Beside code changes, all the newly added parts (`ComponentSpec`, `Executor`,
component interface) need to be accessible in pipeline running environment in
order to run the pipeline properly.
