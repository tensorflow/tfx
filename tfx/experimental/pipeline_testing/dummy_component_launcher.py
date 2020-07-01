# Lint as: python2, python3
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
"""dummy component launcher which launches python executors and dummy executors in process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_executor
from tfx.experimental.pipeline_testing import dummy_executor
from tfx.orchestration.launcher import in_process_component_launcher

class DummyComponentLauncher(
    in_process_component_launcher.InProcessComponentLauncher):
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
    if component_id not in self.component_map:
      super(DummyComponentLauncher, self)._run_executor(execution_id,
                                                        input_dict,
                                                        output_dict,
                                                        exec_properties)
    else:
      executor_context = base_executor.BaseExecutor.Context(
          beam_pipeline_args=self._beam_pipeline_args,
          tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
          unique_id=str(execution_id))
      executor = self.component_map[component_id](component_id,
                                                  self.test_data_dir,
                                                  executor_context)
      executor.Do(input_dict, output_dict, exec_properties)

def create_dummy_launcher_class(test_data_dir: Text,
                                component_ids: List[Text],
                                component_map:
                                Dict[Text, dummy_executor.BaseDummyExecutor]):
  """Creates a DummyComponentLauncher class
  Args:
    test_data_dir: The directory where pipeline outputs are recorded
      (pipeline_recorder.py)
    component_ids: List of component ids that should be replaced
      with a dummy executor
    component_map: Dictionary holding user-defined dummy executor
  """
  cls = DummyComponentLauncher
  cls.component_map = dict(component_map)
  for component_id in component_ids:
    cls.component_map[component_id] = dummy_executor.BaseDummyExecutor
  cls.test_data_dir = test_data_dir
  return cls
