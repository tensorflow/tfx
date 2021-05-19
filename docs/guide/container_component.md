# Building Container-based components

Container-based components provide the flexibility to integrate code written in
any language into your pipeline, so long as you can execute that code in a
Docker container.

If you are new to TFX pipelines,
[learn more about the core concepts of TFX pipelines](understanding_tfx_pipelines).

## Creating a Container-based Component

Container-based components are backed by containerized command-line programs. If
you already have a container image, you can use TFX to create a component from
it by using the
[`create_container_component` function](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/container_component.py){: .external }
to declare inputs and outputs. Function parameters:

*   **name:** The name of the component.
*   **inputs:** A dictionary that maps input names to types. outputs: A
    dictionary that maps output names to types parameters: A dictionary that
    maps parameter names to types.
*   **image:** Container image name, and optionally image tag.
*   **command:** Container entrypoint command line. Not executed within a shell.
    The command line can use placeholder objects that are replaced at
    compilation time with the input, output, or parameter. The placeholder
    objects can be imported from
    [`tfx.dsl.component.experimental.placeholders`](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }.
    Note that Jinja templates are not supported.

**Return value:** a Component class inheriting from base_component.BaseComponent
which can be instantiated and used inside the pipeline.

### Placeholders

For a component that has inputs or outputs, the `command` often needs to have
placeholders that are replaced with actual data at runtime. Several placeholders
are provided for this purpose:

*   `InputValuePlaceholder`: A placeholder for the value of the input artifact.
    At runtime, this placeholder is replaced with the string representation of
    the artifact's value.

*   `InputUriPlaceholder`: A placeholder for the URI of the input artifact
    argument. At runtime, this placeholder is replaced with the URI of the input
    artifact's data.

*   `OutputUriPlaceholder`: A placeholder for the URI of the output artifact
    argument. At runtime, this placeholder is replaced with the URI where the
    component should store the output artifact's data.

Learn more about
[TFX component command-line placeholders](https://github.com/tensorflow/tfx/blob/master/tfx/dsl/component/experimental/placeholders.py){: .external }.

### Example Container-based Component

The following is an example of a non-python component that downloads,
transforms, and uploads the data:

```python
import tfx.v1 as tfx

grep_component = tfx.dsl.components.create_container_component(
    name='FilterWithGrep',
    inputs={
        'text': tfx.standard_artifacts.ExternalArtifact,
    },
    outputs={
        'filtered_text': tfx.standard_artifacts.ExternalArtifact,
    },
    parameters={
        'pattern': str,
    },
    # The component code uses gsutil to upload the data to Google Cloud Storage, so the
    # container image needs to have gsutil installed and configured.
    image='google/cloud-sdk:278.0.0',
    command=[
        'sh', '-exc',
        '''
          pattern="$1"
          text_uri="$3"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          text_path=$(mktemp)
          filtered_text_uri="$5"/data  # Adding suffix, because currently the URI are "directories". This will be fixed soon.
          filtered_text_path=$(mktemp)

          # Getting data into the container
          gsutil cp "$text_uri" "$text_path"

          # Running the main code
          grep "$pattern" "$text_path" >"$filtered_text_path"

          # Getting data out of the container
          gsutil cp "$filtered_text_path" "$filtered_text_uri"
        ''',
        '--pattern', tfx.dsl.placeholders.InputValuePlaceholder('pattern'),
        '--text', tfx.dsl.placeholders.InputUriPlaceholder('text'),
        '--filtered-text', tfx.dsl.placeholders.OutputUriPlaceholder('filtered_text'),
    ],
)
```
