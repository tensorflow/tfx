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
from typing import Any, Dict, List, Text, Union, Tuple, cast

import absl
import apache_beam as beam
import tensorflow as tf
from tensorflow_data_validation.api import stats_api
from tensorflow_data_validation.statistics import stats_options as options
from tfx_bsl.tfxio import tf_example_record

from tfx import types
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import json_utils

SERIALIZED_SUBGRAPH = 'serialized_subgraph'
BEAM_PIPELINE_ARGS = 'beam_pipeline_args'
PIPELINE_ROOT = 'pipeline_root'
CHANNEL_MAP = 'channel_map'


class Executor(base_executor.BaseExecutor):

  def _populate_component_dicts(self, input_dict, output_dict, exec_properties):
    self.component_input_dicts = {}
    self.component_output_dicts = {}
    self.component_exec_properties = {}

    for k, v in input_dict.items():
      component_id, input= k.split('_INPUT_')

      if not component_id in self.component_input_dicts:
        self.component_input_dicts[component_id] = {}
      self.component_input_dicts[component_id][input] = v.get()

    for k, v in output_dict.items():
      component_id, output = k.split('_OUTPUT_')

      if not component_id in self.component_output_dicts:
        self.component_output_dicts[component_id] = {}
      self.component_output_dicts[component_id][output] = v.get()

    for k, v in exec_properties.items():
      if '_PARAMETER_' not in k:
        continue

      component_id, parameter = k.split('_PARAMETER_')

      if not component_id in self.component_exec_properties:
        self.component_exec_properties[component_id] = {}
      self.component_exec_properties[component_id][parameter] = v

  def _get_component_executor(self, component, execution_id):
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self.beam_pipeline_args,
        tmp_dir=os.path.join(self.pipeline_root, '.temp', ''),
        unique_id=str(execution_id))

    executor_class_spec = cast(executor_spec.ExecutorClassSpec,
                               component.executor_spec)

    executor = executor_class_spec.executor_class(executor_context)
    return executor

  def _have_matching_beam_io_signatures(self, child, parent):
    child_input_dict = self.component_input_dicts[child.id]
    child_output_dict = self.component_output_dicts[child.id]
    child_exec_properties = self.component_exec_properties[child.id]
    child_executor = self._get_component_executor(child, -1)

    parent_input_dict = self.component_input_dicts[parent.id]
    parent_output_dict = self.component_output_dicts[parent.id]
    parent_exec_properties = self.component_exec_properties[parent.id]
    parent_executor = self._get_component_executor(parent, -1)

    child_input_sig, _ = child_executor.beam_io_signature(child_input_dict, child_output_dict, child_exec_properties)
    _, parent_output_sig = parent_executor.beam_io_signature(parent_input_dict, parent_output_dict, parent_exec_properties)

    return child_input_sig == parent_output_sig

  def _get_fuseable_parents(self, exec_properties, component_id_map):
    channel_map = json.loads(exec_properties[CHANNEL_MAP])
    fuseable_parents = {}

    for k, v in channel_map.items():
      child_id, input_key = k.split('_INPUT_CHANNEL_')
      parent_id, output_key  = v.split('_OUTPUT_CHANNEL_')

      child = component_id_map[child_id]
      parent = component_id_map[parent_id]

      if self._have_matching_beam_io_signatures(child, parent):
        fuseable_parents[child] = parent

    return fuseable_parents

  def _deserialize_components(self, exec_properties):
    serialized_components = json.loads(exec_properties[SERIALIZED_SUBGRAPH])
    # We memoize a mapping of component.id to component for future use in
    # _deserialize_channel_map() and _run_subgraph()
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
    self._populate_component_dicts(input_dict, output_dict, exec_properties)
    components, component_id_map = self._deserialize_components(exec_properties)
    fuseable_parents = self._get_fuseable_parents(exec_properties, component_id_map)

    p = None
    beam_outputs_cache = {}
    beam_inputs_debug = []

    for i, component in enumerate(components):

      curr_input_dict = self.component_input_dicts[component.id]
      curr_output_dict = self.component_output_dicts[component.id]
      curr_exec_properties = self.component_exec_properties[component.id]
      executor = self._get_component_executor(component, i)

      if not p:
        p = executor._make_beam_pipeline()

      if component in fuseable_parents:
        beam_inputs = beam_outputs_cache[fuseable_parents[component]]
      else:
        beam_inputs = executor.read_inputs(
           p, curr_input_dict, curr_output_dict, curr_exec_properties)

      # beam_inputs = executor.read_inputs(
      #     p, curr_input_dict, curr_output_dict, curr_exec_properties)

      beam_outputs = executor.run_component(
          p, beam_inputs, curr_input_dict, curr_output_dict, curr_exec_properties)

      if component in fuseable_parents.values():
        beam_outputs_cache[component] = beam_outputs

      executor.write_outputs(
         p, beam_outputs,  curr_input_dict, curr_output_dict, curr_exec_properties)

      result = p.run()
      result.wait_until_finish()

    print("\n")
    print(fuseable_parents)
    print("\n")
    print(beam_outputs_cache)
