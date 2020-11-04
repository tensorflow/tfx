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

import copy
import os

from typing import Any, Dict, List, Text, cast

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration.config import base_component_config
from tfx.orchestration.launcher import base_component_launcher


class InProcessComponentLauncher(base_component_launcher.BaseComponentLauncher):
  """Responsible for launching a python executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """

  @classmethod
  def can_launch(
      cls, component_executor_spec: executor_spec.ExecutorSpec,
      component_config: base_component_config.BaseComponentConfig) -> bool:
    """Checks if the launcher can launch the executor spec."""
    if component_config:
      return False

    return isinstance(component_executor_spec, executor_spec.ExecutorClassSpec)

  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._beam_pipeline_args,
        tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))

    executor_class_spec = cast(executor_spec.ExecutorClassSpec,
                               self._component_executor_spec)

    # Type hint of component will cause not-instantiable error as
    # component.executor is Type[BaseExecutor] which has an abstract function.
    executor = executor_class_spec.executor_class(
        executor_context)  # type: ignore

    # Make a deep copy for input_dict and exec_properties, because they should
    # be immutable in this context.
    # output_dict can still be changed, specifically properties.
    executor.Do(
        copy.deepcopy(input_dict), output_dict, copy.deepcopy(exec_properties))
