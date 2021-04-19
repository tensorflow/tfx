# Artifact type YAML schema

This directory includes YAML schema of various artifact types defined in TFX
DSL.

Currently, all user-defined artifact types by deriving the `artifact.Artifact`
Python class will be associated with `Artifact.yaml`, with the class import path
encoded as the title. The artifact types in
`tfx.types.standard_artifacts` and `tfx.types.experimental.simple_artifacts`
will be associated with `tfx.{TypeName}.yaml`, where `{TypeName}` is the Python
type name of the artifact type.

Under the hood, Kubeflow Pipelines backend will capture and populate the
artifact metadata according to the YAML schema when IR-based execution is ready.

# Structure of a type YAML schema
At the top level, the artifact type YAML spec should contains three fields:

- **title**: The identifier of the schema. For all the TFX first party artifact
types, the title is `tfx.{TypeName}`. For example, `tfx.Model` and
`tfx.Examples`. For custom artifact types this will be the class import path.

- **type**: For artifact types, this field should be set to `object`.

- **properties**: A mapping from property names to its type and description,
where property type can be one of `int`, `float`, `string`, and description is
a string literal to document the purpose of the property. For first party TFX
artifact types, the property schema mirrors what was defined in
`tfx.types.standard_artifacts`. This field is omitted if it's empty.

# Tests
Currently in TFX Kubeflow V2 runner and its associated compilation logic,
the artifact type YAML schema is inlined in the job payload by
setting `instance_schema` field. Therefore, if the YAML schema changes it's
highly likely that the golden files under `testdata` directory will need to
change as well.

# Type versioning and evolution.
For now, the artifact type ontology is published with the SDK, which means if
there are multiple pipelines authored by different SDK versions running on the
same same GCP project (which, shares the same MLMD instance under the hood),
there might be type schema incompatibility.

We're working towards solving this issue by moving the type ontology out of the
SDK.
