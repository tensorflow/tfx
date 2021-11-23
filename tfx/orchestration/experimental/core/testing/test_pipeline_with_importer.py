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
"""Pipeline with an importer node for testing."""

from tfx.dsl.compiler import compiler
from tfx.dsl.components.common import importer
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Creates a pipeline with an importer node for testing."""
  inode = importer.Importer(
      source_uri='my_url',
      reimport=True,
      custom_properties={
          'int_custom_property': 123,
          'str_custom_property': 'abc',
      },
      artifact_type=standard_artifacts.Schema).with_id('my_importer')
  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[inode],
      execution_mode=pipeline_lib.ExecutionMode.SYNC)
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
