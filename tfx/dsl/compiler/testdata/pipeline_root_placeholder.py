# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Test pipeline for tfx.dsl.compiler.compiler."""
import os

from tfx.components import CsvExampleGen
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import pipeline


def create_test_pipeline():
  """Builds a pipeline with Placeholder in pipeline_root."""
  pipeline_name = "pipeline_root_placeholder"
  tfx_root = "tfx_root"
  data_path = os.path.join(tfx_root, "data_path")
  pipeline_root = ph.runtime_info("platform_config").base_dir

  example_gen = CsvExampleGen(input_base=data_path)
  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[example_gen],
      enable_cache=True,
      execution_mode=pipeline.ExecutionMode.SYNC)
