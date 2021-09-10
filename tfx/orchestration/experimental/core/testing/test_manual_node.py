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
"""Test pipeline with only manual node."""

from tfx.dsl.compiler import compiler
from tfx.dsl.components.common import manual_node
from tfx.orchestration import pipeline as pipeline_lib
from tfx.proto.orchestration import pipeline_pb2


def create_pipeline() -> pipeline_pb2.Pipeline:
  """Builds a test pipeline with only manual node."""
  manual = manual_node.ManualNode(description='Do something.')

  pipeline = pipeline_lib.Pipeline(
      pipeline_name='my_pipeline',
      pipeline_root='/path/to/root',
      components=[
          manual
      ],
      enable_cache=True)
  dsl_compiler = compiler.Compiler()
  return dsl_compiler.compile(pipeline)
