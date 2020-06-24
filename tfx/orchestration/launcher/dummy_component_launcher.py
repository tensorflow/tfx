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
from typing import Any, Dict, List, Optional, Text
import absl

from tfx import types
from tfx.components.base import base_node
from tfx.components.base import base_executor
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.config import base_component_config
from tfx.orchestration.launcher.in_process_component_launcher import InProcessComponentLauncher
from tfx.experimental.dummy_executor import BaseDummyExecutor

class BaseDummyComponentLauncher(InProcessComponentLauncher):
  """Responsible for launching a dummy executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """
  def __init__(self,
               component: base_node.BaseNode,
               pipeline_info: data_types.PipelineInfo,
               driver_args: data_types.DriverArgs,
               metadata_connection: metadata.Metadata,
               beam_pipeline_args: List[Text],
               additional_pipeline_args: Dict[Text, Any],
               component_config: Optional[
                   base_component_config.BaseComponentConfig] = None):
    super(BaseDummyComponentLauncher, self).__init__(component,
                                                     pipeline_info,
                                                     driver_args,
                                                     metadata_connection,
                                                     beam_pipeline_args,
                                                     additional_pipeline_args,
                                                     component_config)
    self._record_dir = os.path.join(os.environ['HOME'],
                                    'tfx/tfx/examples/chicago_taxi_pipeline',
                                    'testdata')
    self._dummy_executors = {}
    if not os.path.exists(self._record_dir):
      raise Exception("Must record input/output first")


  def set_dummy_executors(self, component_ids: List[Text]):
    """
    component_ids: list of component_id to set a dummy executor
    """
    for component_id in component_ids:
      self._dummy_executors[component_id] = BaseDummyExecutor

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    component_id = self._component_info.component_id
    if component_id not in self._dummy_executors.keys():
      super(BaseDummyComponentLauncher, self)._run_executor(execution_id,
                                                            input_dict,
                                                            output_dict,
                                                            exec_properties)
      # TODO verification for user's executor
    else:
      executor_context = base_executor.BaseExecutor.Context(
          beam_pipeline_args=self._beam_pipeline_args,
          tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
          unique_id=str(execution_id))
      executor = self._dummy_executors[component_id](component_id,
                                                     self._record_dir,
                                                     executor_context)
      executor.Do(input_dict, output_dict, exec_properties)


class MyDummyComponentLauncher(BaseDummyComponentLauncher):
  """
  EX> creating a dummy component launcher
  """
  def __init__(self,
               component: base_node.BaseNode,
               pipeline_info: data_types.PipelineInfo,
               driver_args: data_types.DriverArgs,
               metadata_connection: metadata.Metadata,
               beam_pipeline_args: List[Text],
               additional_pipeline_args: Dict[Text, Any],
               component_config: Optional[
                   base_component_config.BaseComponentConfig] = None):
    absl.logging.info("Launching MyDummyComponentLauncher")
    super(MyDummyComponentLauncher, self).__init__(component,
                                                   pipeline_info,
                                                   driver_args,
                                                   metadata_connection,
                                                   beam_pipeline_args,
                                                   additional_pipeline_args,
                                                   component_config)
    self.set_dummy_executors(['CsvExampleGen', 'StatisticsGen', 'SchemaGen', \
                              'ExampleValidator', 'Transform', \
                              'Evaluator', 'Pusher'])
