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
"""Common data types for orchestration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text, Type, Union
from absl import logging

from tfx import types
from tfx.utils import json_utils

# Regex pattern of RuntimeParameter.
# Use \\* to deal with escaping in json-serialized version of objects.
RUNTIME_PARAMETER_PATTERN = (r'({\\*"__class__\\*": \\*"RuntimeParameter\\*", '
                             r'.*?})')

PARAMETER_NAME_LITERAL = r'(\\*"RuntimeParameter\\*")'


class ExecutionDecision(object):
  """ExecutionDecision records how executor should perform next execution.

  Attributes:
    input_dict: Updated key -> types.Artifact for inputs that will be used by
      actual execution.
    output_dict: Updated key -> types.Artifact for outputs that will be used by
      actual execution.
    exec_properties: Updated dict of other execution properties that will be
      used by actual execution.
    execution_id: Registered execution_id for the upcoming execution.
    use_cached_results: Whether or not to use a cached result.
  """

  def __init__(self,
               input_dict: Dict[Text, List[types.Artifact]],
               output_dict: Dict[Text, List[types.Artifact]],
               exec_properties: Dict[Text, Any],
               execution_id: int = None,
               use_cached_results: Optional[bool] = False):
    self.input_dict = input_dict
    self.output_dict = output_dict
    self.exec_properties = exec_properties
    self.execution_id = execution_id
    self.use_cached_results = use_cached_results


class ExecutionInfo(object):
  """ExecutionInfo contains information populated during execution phase.

  Attributes:
    input_dict: Updated key -> List of types.Artifact for inputs that was used
      during the actual execution.
    output_dict: Updated key -> List of types.Artifact for outputs that was
      generated during the actual execution.
    exec_properties: execution properties used in this execution.
    execution_id: Registered execution_id for the execution.
  """

  def __init__(self, input_dict: Dict[Text, List[types.Artifact]],
               output_dict: Dict[Text, List[types.Artifact]],
               exec_properties: Dict[Text, Any], execution_id: int):
    self.input_dict = input_dict
    self.output_dict = output_dict
    self.exec_properties = exec_properties
    self.execution_id = execution_id


class DriverArgs(object):
  """Args to driver from orchestration system.

  Attributes:
    enable_cache: whether cache is enabled in current execution.
    interactive_resolution: whether to skip MLMD channel artifact resolution, if
      artifacts are already resolved for a channel when running in interactive
      mode.
  """

  def __init__(self,
               enable_cache: bool = True,
               interactive_resolution: bool = False):
    self.enable_cache = enable_cache
    self.interactive_resolution = interactive_resolution


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

  def __repr__(self):
    return ('PipelineInfo('
            'pipeline_name: %s, '
            'pipeline_root: %s, '
            'run_id: %s)') % (self.pipeline_name, self.pipeline_root,
                              self.run_id)

  @property
  def pipeline_run_context_name(self) -> Text:
    """Context name for the current pipeline run."""
    return '{}.{}'.format(self.pipeline_name, self.run_id)

  @property
  def pipeline_context_name(self) -> Text:
    """Context name for the pipeline."""
    return self.pipeline_name


class ComponentInfo(object):
  """Component info.

  Attributes:
    component_type: type of the component. Usually determined by the executor
      python path or image uri of.
    component_id: a unique identifier of the component instance within pipeline.
    pipeline_info: the pipeline info of the current pipeline run.
  """

  def __init__(self, component_type: Text, component_id: Text,
               pipeline_info: PipelineInfo):
    self.component_type = component_type
    self.component_id = component_id
    self.pipeline_info = pipeline_info

  def __repr__(self):
    return ('ComponentInfo('
            'component_type: %s, '
            'component_id: %s, '
            'pipeline_info: %s)') % (self.component_type, self.component_id,
                                     self.pipeline_info)

  @property
  def component_run_context_name(self) -> Text:
    """"Context name for current component run."""
    if self.pipeline_info.run_id:
      return '{}.{}'.format(self.pipeline_info.pipeline_run_context_name,
                            self.component_id)
    else:
      return '{}.{}'.format(self.pipeline_info.pipeline_context_name,
                            self.component_id)


# TODO(b/146361011): Implement a checking mechanism preventing users from using
# RuntimeParameter in DAG runner other than Kubeflow Pipelines.
class RuntimeParameter(json_utils.Jsonable):
  """Runtime parameter.

  Currently only supported on KubeflowDagRunner.

  Attributes:
    name: The name of the runtime parameter.
    default: Default value for runtime params when it's not explicitly
      specified.
    ptype: The type of the runtime parameter.
    description: Description of the usage of the parameter.
  """

  def __init__(
      self,
      name: Text,
      ptype: Type = None,  # pylint: disable=g-bare-generic
      default: Optional[Union[int, float, Text]] = None,
      description: Optional[Text] = None):
    logging.warn('RuntimeParameter is only supported on Cloud-based DAG '
                 'runner currently.')

    if ptype and ptype not in [int, float, Text]:
      raise RuntimeError('Only str and scalar runtime parameters are supported')
    if (default and ptype) and not isinstance(default, ptype):
      raise TypeError('Default value must be consistent with specified ptype')
    self.name = name
    self.default = default
    self.ptype = ptype
    self.description = description

  def __repr__(self):
    """Easily convert RuntimeParameter to str.

    This provides a unified way to call str(x) when x can be either str or
    RuntimeParameter. Note: if ptype == Text or None, the serialization will be
    wrapped in double quotes.

    Returns:
      The json serialized version of RuntimeParameter.
    """
    return json_utils.dumps(self)

  def __eq__(self, other):
    return (isinstance(other.__class__, self.__class__) and
            self.name == other.name and self.default == other.default and
            self.ptype == other.ptype and self.description == other.description)

  def __hash__(self):
    """RuntimeParameter is uniquely identified by its name."""
    return self.name.__hash__()
