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
"""Common data types for orchestration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text

from tfx.utils import types


class ExecutionDecision(object):
  """ExecutionDecision records how executor should perform next execution.

  Attributes:
    input_dict: Updated key -> TfxArtifact for inputs that will be used by
      actual execution.
    output_dict: Updated key -> TfxArtifact for outputs that will be used by
      actual execution.
    exec_properties: Updated dict of other execution properties that will be
      used by actual execution.
    execution_id: Registered execution_id for the upcoming execution. If None,
      then no execution needed.
  """

  def __init__(self,
               input_dict: Dict[Text, List[types.TfxArtifact]],
               output_dict: Dict[Text, List[types.TfxArtifact]],
               exec_properties: Dict[Text, Any],
               execution_id: Optional[int] = None):
    self.input_dict = input_dict
    self.output_dict = output_dict
    self.exec_properties = exec_properties
    self.execution_id = execution_id


class DriverArgs(object):
  """Args to driver from orchestration system.

  Attributes:
    worker_name: orchestrator specific instance name for the worker running
      current component.
    base_output_dir: common base directory shared by all components in current
      pipeline execution.
    enable_cache: whether cache is enabled in current execution.
  """

  def __init__(self, worker_name: Text, base_output_dir: Text,
               enable_cache: bool):
    self.worker_name = worker_name
    self.base_output_dir = base_output_dir
    self.enable_cache = enable_cache
