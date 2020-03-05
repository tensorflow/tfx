# The ImporterNode

An `ImporterNode` is a special TFX node. Unlike other component nodes, it does
not process or generate any data or model. Instead, it takes a uri of some
existing source and registers that in MLMD so that downstream components can
take them as input artifacts.

An `ImporterNode` takes the following arguments:

*   `instance_name`: The name of the `ImporterNode` instance.
*   `source_uri`: The uri to the existing source to be imported in MLMD as
    artifact
*   `artifact_type`: The type of the artifact to register
*   `reimport`: A bool value indicating whether to reimport the artifact if the
    same uri has already been registered in MLMD. If this value is set to
    `True`, `ImporterNode` will always import the source_uri into a new
    artifact. If set to`False`(default),`ImporterNod` will reuse the latest
    existing artifact that has been registered in MLMD.
*   `properties`: A dict of MLMD properties to attach to the artifact to be
    registered
*   `custom_properties`: A dict of MLMD custom properties to attach to the
    artifact to be registered

## Using the ImporterNode

To define an `ImporterNode`, typical code looks like the following:

```Python
from tfx.components import ImporterNode

_user_schema_path = ...

user_schema_importer = ImporterNode(
    instance_name='import_user_schema',
    source_uri=_user_schema_path,
    artifact_type=Schema)
```

Using the output of an `ImporterNode` is the same as using the output of any
component node. For example, if we want to use the result of
the`user_schema_importer`instance above, we only need to get its output with
key`result`:

```Python
validate_stats = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=user_schema_importer.outputs['result'])
```
