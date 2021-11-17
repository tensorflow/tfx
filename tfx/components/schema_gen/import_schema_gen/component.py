# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements a special schema gen variant that import a pre-defined schema."""

from tfx import types
from tfx.components.schema_gen.import_schema_gen import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class ImportSchemaGen(base_component.BaseComponent):
  """A TFX ImportSchemaGen component to import a schema file into the pipeline.

  ImportSchemaGen is a specialized SchemaGen which imports a pre-defined schema
  file into the pipeline.

  In a typical TFX pipeline, users are expected to review the schemas generated
  with `SchemaGen` and store them in SCM or equivalent. Those schema files can
  be brought back to pipelines using ImportSchemaGen.

  Here is an example to use the ImportSchemaGen:

  ```
  schema_gen = ImportSchemaGen(schema_file=schema_path)
  ```

  Component `outputs` contains:
   - `schema`: Channel of type `standard_artifacts.Schema` for schema result.

  See [the SchemaGen guide](https://www.tensorflow.org/tfx/guide/schemagen)
  for more details.

  ImportSchemaGen works almost similar to `Importer` except following:
  - `schema_file` should be the full file path instead of directory holding it.
  - `schema_file` is copied to the output artifact. This is different from
    `Importer` that loads an "Artifact" by setting its URI to the given path.
  """

  SPEC_CLASS = standard_component_specs.ImportSchemaGenSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self, schema_file: str):
    """Init function for the ImportSchemaGen.

    Args:
      schema_file: File path to the input schema file. This file will be copied
        to the output artifact which is generated inside the pipeline root
        directory.
    """
    spec = standard_component_specs.ImportSchemaGenSpec(
        schema_file=schema_file,
        schema=types.Channel(type=standard_artifacts.Schema))
    super().__init__(spec=spec)
