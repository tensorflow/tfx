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
"""dummy component launcher which launches python executors and dummy executors in process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import abc
from typing import Any, Dict, List, Text
import absl

from tfx import types
from tfx.components.base import base_executor
from tfx.orchestration.launcher.in_process_component_launcher import InProcessComponentLauncher
from tfx.experimental.pipeline_testing.dummy_executor import BaseDummyExecutor

class BaseDummyComponentLauncher(InProcessComponentLauncher):
  """Responsible for launching a dummy executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """
  def __init__(self, **kwargs):
    super(BaseDummyComponentLauncher, self).__init__(**kwargs)
    self._component_map = {}
    self._component_ids = []

  @abc.abstractmethod
  def set_dummy_executors(self,
                          record_dir: Text,
                          component_ids: List[Text],
                          component_map:
                          Dict[Text, base_executor.BaseExecutor] = None):
    pass

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    component_id = self._component_info.component_id

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._beam_pipeline_args,
        tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))
    if component_id in self._component_ids:
      executor = BaseDummyExecutor(component_id,
                                   self._record_dir,
                                   executor_context)
      executor.Do(input_dict, output_dict, exec_properties)
    elif component_id in self._component_map.keys():
      executor = self._component_map[component_id](executor_context)
      executor.Do(input_dict, output_dict, exec_properties)
    else:
      super(BaseDummyComponentLauncher, self)._run_executor(execution_id,
                                                            input_dict,
                                                            output_dict,
                                                            exec_properties)
class MyDummyComponentLauncher(BaseDummyComponentLauncher):
  """
  Concrete dummy component launcher
  """
  def __init__(self, **kwargs):
    absl.logging.info("Launching MyDummyComponentLauncher")
    super(MyDummyComponentLauncher, self).__init__(**kwargs)
    record_dir = os.path.join(os.environ['HOME'],
                              'tfx/tfx/experimental/pipeline_testing/',
                              'examples/chicago_taxi_pipeline/testdata')
    self.set_dummy_executors(record_dir, component_ids=['CsvExampleGen', \
                              'StatisticsGen', 'SchemaGen', \
                              'ExampleValidator', 'Transform', \
                              'Evaluator', 'Pusher'])
  def set_dummy_executors(self,
                          record_dir: Text,
                          component_ids: List[Text],
                          component_map:
                          Dict[Text, base_executor.BaseExecutor] = None):
    """
    component_ids: list of component_id to set a dummy executor
    component_map: dictionary holding user custom executors
    """
    # TODO: if component id in both component_ids and component_map?
    self._record_dir = record_dir
    for component_id in component_ids:
      self._component_ids.append(component_id)
    if component_map:
      for component_id, custom_executor in component_map.items():
        self._component_map[component_id] = custom_executor
