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
"""Functions for creating container components from kubeflow components."""

from typing import Any, Callable, Dict

from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.extensions.experimental.kfp_compatibility.proto import kfp_component_spec_pb2
from tfx.types import standard_artifacts
import yaml

from google.protobuf import json_format


def load_kfp_yaml_container_component(
    path: str) -> Callable[..., base_component.BaseComponent]:
  """Creates a container-based component from a Kubeflow component spec.

  See
  https://www.kubeflow.org/docs/pipelines/reference/component-spec/

  Example:
    component = load_kfp_yaml_container_component(
      "kfp_pipelines_root/components/datasets/Chicago_Taxi_Trips/component.yaml"
    )

  Args:
    path: local file path of a Kubeflow Pipelines component YAML file.

  Returns:
    Container component that can be instantiated in a TFX pipeline.
  """
  with open(path) as component_file:
    data = yaml.load(component_file, Loader=yaml.SafeLoader)
  _convert_target_fields_to_kv_pair(data)
  component_spec = json_format.ParseDict(data,
                                         kfp_component_spec_pb2.ComponentSpec())
  container = component_spec.implementation.container
  command = (
      list(map(_get_command_line_argument_type, container.command)) +
      list(map(_get_command_line_argument_type, container.args)))
  # TODO(ericlege): Support classname to class translation in inputs.type
  inputs = {
      item.name: standard_artifacts.String for item in component_spec.inputs
  }
  outputs = {
      item.name: standard_artifacts.String for item in component_spec.outputs
  }
  parameters = {}
  return container_component.create_container_component(
      name=component_spec.name,
      image=container.image,
      command=command,
      inputs=inputs,
      outputs=outputs,
      parameters=parameters,
  )


def _convert_target_fields_to_kv_pair(parsed_dict: Dict[str, Any]) -> None:
  """Converts in place specific string fields to key value pairs of {constantValue: [Text]} for proto3 compatibility.

  Args:
    parsed_dict: dictionary obtained from parsing a Kubeflow component spec.
      This argument is modified in place.

  Returns:
    None
  """
  conversion_string_paths = [
      ['implementation', 'container', 'command'],
      ['implementation', 'container', 'args'],
  ]
  for path in conversion_string_paths:
    parsed_dict_location = parsed_dict
    for label in path:
      parsed_dict_location = parsed_dict_location.get(label, {})
    if isinstance(parsed_dict_location, list):
      for ind, value in enumerate(parsed_dict_location):
        if isinstance(value, str):
          parsed_dict_location[ind] = {'constantValue': value}


def _get_command_line_argument_type(
    command: kfp_component_spec_pb2.StringOrPlaceholder
) -> placeholders.CommandlineArgumentType:
  """Converts a container command to the corresponding type.

  Args:
    command: StringOrPlaceholder which encodes a container command.

  Returns:
    command to be passed into create_container_component.
  """
  if command.HasField('constantValue'):
    return command.constantValue
  if command.HasField('inputValue'):
    return placeholders.InputValuePlaceholder(command.inputValue)
  if command.HasField('inputPath'):
    return placeholders.InputUriPlaceholder(command.inputPath)
  if command.HasField('outputPath'):
    return placeholders.OutputUriPlaceholder(command.outputPath)
  raise ValueError('Unrecognized command %s' % command)
