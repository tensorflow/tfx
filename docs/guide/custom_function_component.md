# Custom Python function components

Python function-based component definition makes it easier for you to create TFX
custom components, by saving you the effort of defining a component
specification class, executor class, and component interface class. In this
component definition style, you write a function that is annotated with type
hints. The type hints describe the input artifacts, output artifacts, and
parameters of your component.

Writing your custom component in this style is very straightforward, as in the
following example.

```python
@component
def MyValidationComponent(
    model: InputArtifact[Model],
    blessing: OutputArtifact[Model],
    accuracy_threshold: Parameter[int] = 10,
    ) -> OutputDict(accuracy=float):
  '''My simple custom model validation component.'''

  accuracy = evaluate_model(model)
  if accuracy >= accuracy_threshold:
    write_output_blessing(blessing)

  return {
    'accuracy': accuracy
  }
```

Under the hood, this defines a custom component that is a subclass of
[`BaseComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_component.py){: .external }
and its Spec and Executor classes.

Note: the feature (BaseBeamComponent based component by annotating a function
with `@component(use_beam=True)`) described below is experimental and there is
no public backwards compatibility guarantees.

If you want to define a subclass of
[`BaseBeamComponent`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_component.py){: .external }
such that you could use a beam pipeline with TFX-pipeline-wise shared
configuration, i.e., `beam_pipeline_args` when compiling the pipeline
([Chicago Taxi Pipeline Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_simple.py#L192){: .external })
you could set `use_beam=True` in the decorator and add another
`BeamComponentParameter` with default value `None` in your function as the
following example:

```python
@component(use_beam=True)
def MyDataProcessor(
    examples: InputArtifact[Example],
    processed_examples: OutputArtifact[Example],
    beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
    ) -> None:
  '''My simple custom model validation component.'''

  with beam_pipeline as p:
    # data pipeline definition with beam_pipeline begins
    ...
    # data pipeline definition with beam_pipeline ends
```

If you are new to TFX pipelines,
[learn more about the core concepts of TFX pipelines](understanding_tfx_pipelines).

## Inputs, outputs, and parameters

In TFX, inputs and outputs are tracked as Artifact objects which describe the
location of and metadata properties associated with the underlying data; this
information is stored in ML Metadata. Artifacts can describe complex data types
or simple data types, such as: int, float, bytes, or unicode strings.

A parameter is an argument (int, float, bytes, or unicode string) to a component
known at pipeline construction time. Parameters are useful for specifying
arguments and hyperparameters like training iteration count, dropout rate, and
other configuration to your component. Parameters are stored as properties of
component executions when tracked in ML Metadata.

Note: Currently, output simple data type values cannot be used as parameters
since they are not known at execution time. Similarly, input simple data type
values currently cannot take concrete values known at pipeline construction
time. We may remove this restriction in a future release of TFX.

## Definition

To create a custom component, write a function that implements your custom logic
and decorate it with the
[`@component` decorator](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/decorators.py){: .external }
from the `tfx.dsl.component.experimental.decorators` module. To define your
component’s input and output schema, annotate your function’s arguments and
return value using annotations from the
`tfx.dsl.component.experimental.annotations` module:

*   For each **artifact input**, apply the `InputArtifact[ArtifactType]` type
    hint annotation. Replace `ArtifactType` with the artifact’s type, which is a
    subclass of `tfx.types.Artifact`. These inputs can be optional arguments.

*   For each **output artifact**, apply the `OutputArtifact[ArtifactType]` type
    hint annotation. Replace `ArtifactType` with the artifact’s type, which is a
    subclass of `tfx.types.Artifact`. Component output artifacts should be
    passed as input arguments of the function, so that your component can write
    outputs to a system-managed location and set appropriate artifact metadata
    properties. This argument can be optional or this argument can be defined
    with a default value.

*   For each **parameter**, use the type hint annotation `Parameter[T]`. Replace
    `T` with the type of the parameter. We currently only support primitive
    python types: `bool`, `int`, `float`, `str`, or `bytes`.

*   For **beam pipeline**, use the type hint annotation
    `BeamComponentParameter[beam.Pipeline]`. Set the default value to be `None`.
    The value `None` will be replaced by an instantiated beam pipeline created
    by `_make_beam_pipeline()` of
    [`BaseBeamExecutor`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/components/base/base_beam_executor.py){: .external }

*   For each **simple data type input** (`int`, `float`, `str` or `bytes`) not
    known at pipeline construction time, use the type hint `T`. Note that in the
    TFX 0.22 release, concrete values cannot be passed at pipeline construction
    time for this type of input (use the `Parameter` annotation instead, as
    described in the previous section). This argument can be optional or this
    argument can be defined with a default value. If your component has simple
    data type outputs (`int`, `float`, `str` or `bytes`), you can return these
    outputs using an `OutputDict` instance. Apply the `OutputDict` type hint as
    your component’s return value.

*   For each **output**, add argument `<output_name>=<T>` to the `OutputDict`
    constructor, where `<output_name>` is the output name and `<T>` is the
    output type, such as: `int`, `float`, `str` or `bytes`.

In the body of your function, input and output artifacts are passed as
`tfx.types.Artifact` objects; you can inspect its `.uri` to get its
system-managed location and read/set any properties. Input parameters and simple
data type inputs are passed as objects of the specified type. Simple data type
outputs should be returned as a dictionary, where the keys are the appropriate
output names and the values are the desired return values.

The completed function component can look like this:

```python
import tfx.v1 as tfx
from tfx.dsl.component.experimental.decorators import component

@component
def MyTrainerComponent(
    training_data: tfx.dsl.components.InputArtifact[tfx.types.standard_artifacts.Examples],
    model: tfx.dsl.components.OutputArtifact[tfx.types.standard_artifacts.Model],
    dropout_hyperparameter: float,
    num_iterations: tfx.dsl.components.Parameter[int] = 10
    ) -> tfx.v1.dsl.components.OutputDict(loss=float, accuracy=float):
  '''My simple trainer component.'''

  records = read_examples(training_data.uri)
  model_obj = train_model(records, num_iterations, dropout_hyperparameter)
  model_obj.write_to(model.uri)

  return {
    'loss': model_obj.loss,
    'accuracy': model_obj.accuracy
  }

# Example usage in a pipeline graph definition:
# ...
trainer = MyTrainerComponent(
    examples=example_gen.outputs['examples'],
    dropout_hyperparameter=other_component.outputs['dropout'],
    num_iterations=1000)
pusher = Pusher(model=trainer.outputs['model'])
# ...
```

The preceding example defines `MyTrainerComponent` as a Python function-based
custom component. This component consumes an `examples` artifact as its input,
and produces a `model` artifact as its output. The component uses the
`artifact_instance.uri` to read or write the artifact at its system-managed
location. The component takes a `num_iterations` input parameter and a
`dropout_hyperparameter` simple data type value, and the component outputs
`loss` and `accuracy` metrics as simple data type output values. The output
`model` artifact is then used by the `Pusher` component.
