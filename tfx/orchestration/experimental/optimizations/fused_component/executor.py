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
"""TFX fused_component executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from typing import Any, Dict, List, Text, Mapping, Tuple, cast

from tfx import types
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.utils import json_utils
from tfx.orchestration.kubeflow.node_wrapper import NodeWrapper

SERIALIZED_SUBGRAPH = 'serialized_subgraph'
BEAM_PIPELINE_ARGS = 'beam_pipeline_args'
PIPELINE_ROOT = 'pipeline_root'
CHANNEL_MAP = 'channel_map'


class Executor(base_executor.BaseExecutor):
  """Executes components in FusedComponent subgraph, performing fusion optimization"""

  def _populate_component_dicts(self,
                                input_dict: Dict[Text, List[types.Artifact]],
                                output_dict: Dict[Text, List[types.Artifact]],
                                exec_properties: Dict[Text, Any],
                                components: List[NodeWrapper]) -> None:
    self.component_input_dicts = {}
    self.component_output_dicts = {}
    self.component_exec_properties = {}

    for component in components:
      self.component_input_dicts[component.id] = {}
      self.component_output_dicts[component.id] = {}
      self.component_exec_properties[component.id] = {}

    for k, v in input_dict.items():
      component_id, input_key = k.split('_INPUT_')
      self.component_input_dicts[component_id][input_key] = v.get()

    for k, v in output_dict.items():
      component_id, output_key = k.split('_OUTPUT_')
      self.component_output_dicts[component_id][output_key] = v.get()

    for k, v in exec_properties.items():
      if '_PARAMETER_' not in k:
        continue

      component_id, parameter_key = k.split('_PARAMETER_')
      self.component_exec_properties[component_id][parameter_key] = v

  def _get_component_executor(self, component: NodeWrapper, execution_id: int
                              ) -> base_executor.FuseableBeamExecutor:
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self.beam_pipeline_args,
        tmp_dir=os.path.join(self.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))

    executor_class_spec = cast(executor_spec.ExecutorClassSpec,
                               component.executor_spec)

    executor = executor_class_spec.executor_class(executor_context)
    return executor

  def _have_matching_beam_io_signatures(self, child: NodeWrapper,
                                        parent: NodeWrapper) -> bool:
    child_input_dict = self.component_input_dicts[child.id]
    child_output_dict = self.component_output_dicts[child.id]
    child_exec_properties = self.component_exec_properties[child.id]
    child_executor = self._get_component_executor(child, -1)

    parent_input_dict = self.component_input_dicts[parent.id]
    parent_output_dict = self.component_output_dicts[parent.id]
    parent_exec_properties = self.component_exec_properties[parent.id]
    parent_executor = self._get_component_executor(parent, -1)

    child_input_sig, _ = child_executor.beam_io_signature(
        child_input_dict, child_output_dict, child_exec_properties)
    _, parent_output_sig = parent_executor.beam_io_signature(
        parent_input_dict, parent_output_dict, parent_exec_properties)
    return child_input_sig == parent_output_sig

  def _get_fusion_map(self, exec_properties: Dict[Text, Any],
                      component_id_map: Mapping[Text, NodeWrapper]
                      ) -> Mapping[NodeWrapper, NodeWrapper]:
    channel_map = json.loads(exec_properties[CHANNEL_MAP])
    fusion_map = {}

    for k, v in channel_map.items():
      child_id, _ = k.split('_INPUT_CHANNEL_')
      parent_id, _ = v.split('_OUTPUT_CHANNEL_')
      child = component_id_map[child_id]
      parent = component_id_map[parent_id]
      fusion_map[child] = parent

    return fusion_map

  def _deserialize_components(self, exec_properties: Dict[Text, Any]
                              ) -> Tuple[List[NodeWrapper],
                                         Mapping[Text, NodeWrapper]]:
    serialized_components = json.loads(exec_properties[SERIALIZED_SUBGRAPH])
    components = []
    component_id_map = {}

    for serialized_component in serialized_components:
      component = json_utils.loads(serialized_component)
      component_id_map[component.id] = component
      components.append(component)

    return components, component_id_map

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Executes the components in the FusedComponent subgraph.

    Args:
      input_dict: INPUTS
        - component.id_input_key: Generic format for entries in input_dict.
          Contains the input_dict values of the FusedComponent subgraph's
          components.
      output_dict: OUTPUTS
        - component.id_output_key: Generic format for entries in output_key.
          Contains the output_dict values of the FusedComponent subgraph's
          components.
      exec_properties: PARAMETERS
        - component.id_parameter_key: Generic format for entries in
          exec_properties. Contains the exec_properties values of the
          FusedComponent subgraph's components.
        - serialized_subgraph: List of serialized components in sorted
          topological order

    Returns:
      None
    """
    self.beam_pipeline_args = json.loads(exec_properties[BEAM_PIPELINE_ARGS])
    self.pipeline_root = exec_properties[PIPELINE_ROOT]
    components, component_id_map = self._deserialize_components(exec_properties)
    self._populate_component_dicts(
        input_dict, output_dict, exec_properties, components)
    fusion_map = self._get_fusion_map(exec_properties, component_id_map)

    p = None
    beam_outputs_cache = {}
    for i, component in enumerate(components):
      curr_input_dict = self.component_input_dicts[component.id]
      curr_output_dict = self.component_output_dicts[component.id]
      curr_exec_properties = self.component_exec_properties[component.id]
      executor = self._get_component_executor(component, i)

      if not p:
        p = executor._make_beam_pipeline() # pylint: disable=protected-access

      use_cached_inputs = False
      if component in fusion_map:
        parent = fusion_map[component]
        if self._have_matching_beam_io_signatures(component, parent):
          use_cached_inputs = True

      if use_cached_inputs:
        beam_inputs = beam_outputs_cache[fusion_map[component]]
      else:
        beam_inputs = executor.read_inputs(
            p, curr_input_dict, curr_output_dict, curr_exec_properties)

      beam_outputs = executor.run_component(
          p, beam_inputs, curr_input_dict, curr_output_dict,
          curr_exec_properties)

      if component in fusion_map.values():
        beam_outputs_cache[component] = beam_outputs

      executor.write_outputs(
          p, beam_outputs, curr_input_dict, curr_output_dict,
          curr_exec_properties)

    result = p.run()
    result.wait_until_finish()
