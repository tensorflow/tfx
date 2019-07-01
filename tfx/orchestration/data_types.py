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
    execution_id: Registered execution_id for the upcoming execution.
    use_cached_results: Whether or not to use a cached result.
  """

  def __init__(
      self,
      input_dict: Dict[Text, List[types.TfxArtifact]],
      output_dict: Dict[Text, List[types.TfxArtifact]],
      exec_properties: Dict[Text, Any],
      # TODO(ruoyu): Make this required once finish Airflow migration.
      execution_id: Optional[int] = None,
      use_cached_results: Optional[bool] = False):
    self.input_dict = input_dict
    self.output_dict = output_dict
    self.exec_properties = exec_properties
    self.execution_id = execution_id
    self.use_cached_results = use_cached_results

  # TODO(ruoyu): Deprecate this in favor of use_cached_results once finishing
  # migration to go/tfx-oss-artifact-passing.
  @property
  def execution_needed(self) -> bool:
    """Indicates whether a new execution is needed.

    Returns:
      true if execution_id exists
      false if execution_id does not exist
    """
    return self.execution_id is not None


class DriverArgs(object):
  """Args to driver from orchestration system.

  Attributes:
    worker_name: orchestrator specific instance name for the worker running
      current component.
    base_output_dir: common base directory shared by all components in current
      pipeline execution.
    enable_cache: whether cache is enabled in current execution.
  """

  def __init__(self,
               enable_cache: bool,
               worker_name: Optional[Text] = '',
               base_output_dir: Optional[Text] = ''):
    # TODO(ruoyu): Remove worker_name and base_output_dir once migration to
    # go/tfx-oss-artifact-passing finishes.
    self.worker_name = worker_name
    self.base_output_dir = base_output_dir
    self.enable_cache = enable_cache


class PipelineInfo(object):
  """Pipeline info from orchestration system.

  Attributes:
    pipeline_name: name of the pipeline. We expect this to be unique for
      different pipelines.
    pipeline_root: root directory of the pipeline. We expect this to be unique
      for different pipelines.
    run_id: optional uuid for a single run of the pipeline.
  """

  def __init__(self,
               pipeline_name: Text,
               pipeline_root: Text,
               run_id: Optional[Text] = None):
    self.pipeline_name = pipeline_name
    self.pipeline_root = pipeline_root
    self.run_id = run_id


class ComponentInfo(object):
  """Component info.

  Attributes:
    component_type: type of the component. Usually determined by the executor
      python path or image uri of.
    component_id: a unique identifier of the component instance within pipeline.
  """

  def __init__(self,
               component_type: Text,
               component_id: Text):
    self.component_type = component_type
    self.component_id = component_id
