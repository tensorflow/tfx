# Lint as: python3
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
"""Stub component launcher which launches component executors and stub executors in process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Text, Type

from tfx import types
from tfx.components.base import base_executor
from tfx.experimental.pipeline_testing import base_stub_executor
from tfx.orchestration.launcher import in_process_component_launcher


class StubComponentLauncher(
    in_process_component_launcher.InProcessComponentLauncher):
  """Responsible for launching a stub executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """

  stubbed_component_map = ...  # type: Dict[Text, Type[base_stub_executor.BaseStubExecutor]]
  test_data_dir = ...  # type: Text

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    component_id = self._component_info.component_id
    if component_id not in self.stubbed_component_map:
      super(StubComponentLauncher,
            self)._run_executor(execution_id, input_dict, output_dict,
                                exec_properties)
    else:
      executor_context = base_executor.BaseExecutor.Context(
          beam_pipeline_args=self._beam_pipeline_args,
          tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
          unique_id=str(execution_id))
      executor = self.stubbed_component_map[component_id](component_id,
                                                          self.test_data_dir,
                                                          executor_context)
      executor.Do(input_dict, output_dict, exec_properties)


def get_stub_launcher_class(
    test_data_dir: Text, stubbed_component_ids: List[Text],
    stubbed_component_map: Dict[Text, Type[base_stub_executor.BaseStubExecutor]]
) -> Type[StubComponentLauncher]:
  """Returns a StubComponentLauncher class.

  Factory method returns a StubComponentLauncher class, which then can be
  supported by the pipeline.
  For example:
    class MyPusherStubExecutor(base_stub_executor.BaseStubExecutor){...}
    class MyTransformStubExecutor(base_stub_executor.BaseStubExecutor){...}

    PipelineConfig(
        supported_launcher_classes=[
            stub_component_launcher.get_stub_launcher_class(
                test_data_dir,
                stubbed_component_ids = ['CsvExampleGen'],
                stubbed_component_map = {
                    'Transform': MyTransformStubExecutor,
                    'Pusher': MyPusherStubExecutor})
        ],
    )
  The method also sets the necessary class variables for the
  StubComponentLauncher class, including stubbed_component_ids and
  stubbed_component_map holding custom executor classes, which
  users may define differently per component.

  Args:
    test_data_dir: The directory where pipeline outputs are recorded
      (pipeline_recorder.py).
    stubbed_component_ids: List of component ids that should be replaced with a
      BaseStubExecutor.
    stubbed_component_map: Dictionary holding user-defined stub executor. These
      user-defined stub executors must inherit from
      base_stub_executor.BaseStubExecutor.

  Returns:
    StubComponentLauncher class holding stub executors.
  """
  cls = StubComponentLauncher
  cls.stubbed_component_map = dict(stubbed_component_map)
  for component_id in stubbed_component_ids:
    cls.stubbed_component_map[component_id] = \
                    base_stub_executor.BaseStubExecutor
  cls.test_data_dir = test_data_dir
  return cls
