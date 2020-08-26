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
"""TFX FusedComponent component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import itertools
from typing import Optional, Text, List

from tfx.utils import json_utils
from tfx.components.base import base_component
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.components.statistics_gen import executor
from tfx.types import standard_artifacts
from tfx.types.channel import Channel
from tfx.types.artifact import Artifact
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_component_specs import FusedComponentSpec
from tfx.orchestration.kubeflow import node_wrapper

SERIALIZED_SUBGRAPH = 'serialized_subgraph'
BEAM_PIPELINE_ARGS = 'beam_pipeline_args'
PIPELINE_ROOT = 'pipeline_root'
CHANNEL_MAP = 'channel_map'


class FusedComponent(base_component.BaseComponent):
  """Official TFX fused component.

  The FusedComponent acts as a wrapper for Apache Beam based components whose
  reads,execution, and writes can be fused to prevent extra file I/O.

  """

  SPEC_CLASS = FusedComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               subgraph: List[base_node.BaseNode],
               beam_pipeline_args: List[Text],
               pipeline_root: Text,
               instance_name: Optional[Text] = None):
    """Construct a FusedComponent.

    Args:
      components: logical components of this FusedComponent, in topological
        order.
      instance_name: Optional name assigned to this specific instance of
        FusedComponent.  Required only if multiple FusedComponents are declared
        in the same pipeline.
    """
    self.subgraph = subgraph
    spec = self._create_component_spec(beam_pipeline_args, pipeline_root)
    super(FusedComponent, self).__init__(
        spec=spec, instance_name=instance_name)

  def _serialize_subgraph(self):
    serialized_subgraph = []

    for component in self.subgraph:
      serialized_subgraph.append(
          json_utils.dumps(node_wrapper.NodeWrapper(component)))

    return json.dumps(serialized_subgraph)

  def _find_channel_dependencies(self):
    channel_map = {}

    # Iterate through the subgraph components' inputs to find out if they map
    # to outputs of other subgraph components
    for child in self.subgraph:
      for input_key, input_channel in child.inputs.items():

        for parent in self.subgraph:
          if parent == child:
            continue

          for output_key, output_channel in parent.outputs.items():
            if input_channel == output_channel:
              key = child.id + '_INPUT_CHANNEL_' + input_key
              value = parent.id + '_OUTPUT_CHANNEL_' + output_key
              channel_map[key] = value

    return json.dumps(channel_map)

  def _create_component_spec(self, beam_pipeline_args, pipeline_root):
    parameters = {}
    inputs = {}
    outputs = {}
    spec_kwargs = {}

    for component in self.subgraph:
      for k, v in component.exec_properties.items():
        key = component.id + '_PARAMETER_' + k
        parameters[key] = ExecutionParameter(type=type(v)) # TODO: explain the bug lol write a TODO component.spec.PARAMETERS[k] #
        spec_kwargs[key] = v
        # special case by checking if proto message https://github.com/tensorflow/tfx/blob/9f8908346b4afe7741040f2d7241a8750a516003/tfx/types/component_spec.py#L225

      for k, v in component.inputs.items():
        key = component.id + '_INPUT_' + k
        inputs[key] = component.spec.INPUTS[k]
        spec_kwargs[key] = v

      for k, v in component.outputs.items():
        key = component.id + '_OUTPUT_' + k
        outputs[key] = component.spec.OUTPUTS[k]
        spec_kwargs[key] = v

    parameters[SERIALIZED_SUBGRAPH] = ExecutionParameter(type=str)
    parameters[BEAM_PIPELINE_ARGS] = ExecutionParameter(type=str)
    parameters[PIPELINE_ROOT] = ExecutionParameter(type=str)
    parameters[CHANNEL_MAP] = ExecutionParameter(type=str)

    spec_kwargs[SERIALIZED_SUBGRAPH] = self._serialize_subgraph()
    spec_kwargs[BEAM_PIPELINE_ARGS] = json.dumps(beam_pipeline_args)
    spec_kwargs[PIPELINE_ROOT] = pipeline_root
    spec_kwargs[CHANNEL_MAP] = self._find_channel_dependencies()

    class DynamicFusedComponentSpec(FusedComponentSpec):
      PARAMETERS = parameters
      INPUTS = inputs
      OUTPUTS = outputs

    spec = DynamicFusedComponentSpec(**spec_kwargs)
    return spec

  def in_subgraph(self, component: base_node.BaseNode):
    return component in self.subgraph

  def get_subgraph(self):
    return self.subgraph
