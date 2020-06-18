# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""In process component launcher which launches python executors in process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_executor
from tfx.orchestration.launcher import in_process_component_launcher
import absl

class MyDummyComponentLauncher(in_process_component_launcher.InProcessComponentLauncher):
  """Responsible for launching a dummy executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """
  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    component_id = self._component_info.component_id
    if component_id not in self.dummy_dict.keys():
      super(MyDummyComponentLauncher, self)._run_executor(execution_id, 
                                                          input_dict, 
                                                          output_dict,
                                                          exec_properties)
    else:
      executor_context = base_executor.BaseExecutor.Context(
          beam_pipeline_args=self._beam_pipeline_args,
          tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
          unique_id=str(execution_id))
      metadata_dir = os.path.join(os.environ['HOME'], 'tfx/metadata/chicago_taxi_beam/meta3.db')
      executor = self.dummy_dict[component_id](component_id, metadata_dir, executor_context)
      absl.logging.info("Running executor [%s]", executor)
      executor.Do(input_dict, output_dict, exec_properties)
