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
"""Executor for ImportSchemaGen."""

import os
from typing import Any, Dict, List

from absl import logging
from tfx import types
from tfx.components.schema_gen import executor as schema_gen_executor
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils


class Executor(base_executor.BaseExecutor):
  """TFX ImportSchemaGen executor."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """ImportSchemaGen executor entrypoint.

    This generate Schema artifact with given schema_file.

    Args:
      input_dict: Should be empty.
      output_dict: Output dict from key to a list of artifacts, including:
        - schema: A list of 'Schema' artifact of size one.
      exec_properties: A dict of execution properties, includes:
        - schema_file: Source schema file path.

    Returns:
      None
    """
    source_file_path = exec_properties.get(
        standard_component_specs.SCHEMA_FILE_KEY)
    if not source_file_path:
      raise ValueError('Schema file path is missing in exec_properties.')
    output_uri = os.path.join(
        artifact_utils.get_single_uri(
            output_dict[standard_component_specs.SCHEMA_KEY]),
        schema_gen_executor.DEFAULT_FILE_NAME)

    # Check whether the input file has a proper schema proto.
    _ = io_utils.SchemaReader().read(source_file_path)

    io_utils.copy_file(source_file_path, output_uri)
    logging.info('Copied a schema file from %s to %s.', source_file_path,
                 output_uri)
