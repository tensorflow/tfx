# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Test sample pipeline with external artifacts for tfx.dsl.compiler.compiler."""

import os

from tfx.components import CsvExampleGen
from tfx.orchestration import pipeline

_pipeline_name = 'external_artifacts_pipeline'
_pipeline_root = os.path.join('pipeline', _pipeline_name)


def create_test_pipeline():
  """Builds an external artifacts example pipeline."""
  tfx_root = 'tfx_root'
  data_path = os.path.join(tfx_root, 'data_path')

  example_gen = CsvExampleGen(input_base=data_path)
  example_gen.outputs['examples'].set_external(
      predefined_artifact_uris=['/external_directory/examples/123']
  )
  return pipeline.Pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      components=[example_gen])
