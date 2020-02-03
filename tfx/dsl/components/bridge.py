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

"""This module provides simplified ways to create components."""

__all__ = [
    'create_component_from_func',
    'create_tfx_component_class_from_spec',
    'enable_new_components',
    'InputPath',
    'OutputPath',
    'InputArtifactUri',
    'OutputArtifactUri',
]

import contextlib
import copy
import json
from typing import Any, Mapping, Type

from kfp import components
from kfp.components import create_component_from_func
from kfp.components import InputPath
from kfp.components import OutputPath
from kfp.components import structures
# from kfp.components._components import _resolve_command_line_and_paths
# from kfp.components._naming import _sanitize_python_function_name

from ml_metadata.proto import metadata_store_pb2
from tfx.components.base.base_component import BaseComponent
from tfx.components.base.executor_spec import ExecutorContainerSpec
from tfx.types import component_spec as tfx_component_spec
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel
from tfx.types.component_spec import ChannelParameter
from tfx.types.standard_artifacts import ANY_ARTIFACT_TYPE_NAME


def _create_artifact_type(type_struct) -> metadata_store_pb2.ArtifactType:
  # TODO(avolkov): Handle schema
  if not type_struct:
    type_name = ANY_ARTIFACT_TYPE_NAME
  elif isinstance(type_struct, str):
    type_name = str(type_struct)
  else:
    type_name = json.dumps(type_struct)
  return metadata_store_pb2.ArtifactType(name=type_name)


def _create_channel_parameter(type_struct) -> ChannelParameter:
  return ChannelParameter(mlmd_artifact_type=_create_artifact_type(type_struct))


def _create_empty_artifact(type_struct) -> Artifact:
  return Artifact(mlmd_artifact_type=_create_artifact_type(type_struct))


def _create_channel_with_empty_artifact(type_struct) -> Channel:
  artifact_type = _create_artifact_type(type_struct)
  return Channel(
      mlmd_artifact_type=artifact_type,
      artifacts=[
          Artifact(artifact_type),
      ],
  )


def type_to_type_struct(typ):
  if not typ:
    return None
  return str(typ.__module__) + '.' + str(typ.__name__)


class InputArtifactUri:
  """InputArtifactUri represents a URI pointing to readable data of certain type."""

  def __init__(self, data_type: Type[Any] = None, type_struct: str = None):
    if type_struct:
      self._data_type_struct = type_struct
    else:
      self._data_type_struct = type_to_type_struct(data_type)

  def to_dict(self):
    properties = {
        'data_type': self._data_type_struct,
    }
    return {'InputArtifactUri': properties}

  @staticmethod
  def get_data_type(type_struct):
    return type_struct['InputArtifactUri']['data_type']

  @staticmethod
  def type_struct_is_input_artifact_uri(type_struct) -> bool:
    return type_struct == 'InputArtifactUri' or isinstance(
        type_struct, dict) and type_struct.keys() == ['InputArtifactUri']


class OutputArtifactUri:
  """OutputArtifactUri represents a URI pointing to writable location for data of sertain type."""

  def __init__(self, data_type: Type[Any] = None, type_struct: str = None):
    if type_struct:
      self._data_type_struct = type_struct
    else:
      self._data_type_struct = type_to_type_struct(data_type)

  def to_dict(self):
    properties = {
        'data_type': self._data_type_struct,
    }
    return {'OutputArtifactUri': properties}

  @staticmethod
  def get_data_type(type_struct):
    return type_struct['OutputArtifactUri']['data_type']

  @staticmethod
  def type_struct_is_output_artifact_uri(type_struct) -> bool:
    return type_struct == 'OutputArtifactUri' or isinstance(
        type_struct, dict) and type_struct.keys() == ['OutputArtifactUri']


class ExecutionProperty:
  """ExecutionProperty represents a primitive input whose values might be queried in the future."""

  def __init__(self, data_type: Type[Any] = None, type_struct: str = None):
    if type_struct:
      self._data_type_struct = type_struct
    else:
      self._data_type_struct = type_to_type_struct(data_type)

  def to_dict(self):
    properties = {
        'data_type': self._data_type_struct,
    }
    return {'QueryableProperty': properties}


def _sanitize_python_class_name(name: str) -> str:
  function_name = components._naming._sanitize_python_function_name(name)  # pylint: disable=protected-access
  return ''.join(word.title() for word in function_name.split('_'))


def create_tfx_component_class_from_spec(
    component_spec: structures.ComponentSpec,
) -> BaseComponent:
  """Generates TFX components based on KFP-style ComponentSpec."""

  container_spec = component_spec.implementation.container

  if container_spec is None:
    raise TypeError(
        'Only components with container implementation can be instantiated in TFX at this moment.'
    )

  input_name_to_python = {
      input_spec.name:
      components._naming._sanitize_python_function_name(input_spec.name)  # pylint: disable=protected-access
      for input_spec in component_spec.inputs or []
  }
  output_name_to_python = {
      output_spec.name:
      components._naming._sanitize_python_function_name(output_spec.name)  # pylint: disable=protected-access
      for output_spec in component_spec.outputs or []
  }

  input_channel_parameters = {}
  output_channel_parameters = {}
  default_input_channels = {}
  output_channels = {}
  execution_parameters = {}
  inputs_that_are_input_artifact_uris = set()
  inputs_that_are_output_artifact_uris = set()
  inputs_that_are_execution_properties = set()

  for input_spec in component_spec.inputs or []:
    python_input_name = input_name_to_python[input_spec.name]

    # Handling special cases like InputArtifactUri, OutputArtifactUri
    if InputArtifactUri.type_struct_is_input_artifact_uri(input_spec.type):
      inputs_that_are_input_artifact_uris.add(input_spec.name)
      fixed_input = copy.copy(input_spec)
      data_type = InputArtifactUri.get_data_type(input_spec.type)
      fixed_input.type = {'Uri': {'data_type': data_type}}

      input_channel_parameters[python_input_name] = _create_channel_parameter(
          data_type)
      if input_spec.optional:
        default_input_channels[python_input_name] = (
            _create_channel_with_empty_artifact(data_type))
      continue

    if OutputArtifactUri.type_struct_is_output_artifact_uri(input_spec.type):
      inputs_that_are_output_artifact_uris.add(input_spec.name)
      fixed_output = copy.copy(input_spec)
      data_type = OutputArtifactUri.get_data_type(input_spec.type)
      fixed_output.type = {'Uri': {'data_type': data_type}}
      output_channel_parameters[python_input_name] = _create_channel_parameter(
          data_type)
      output_channels[python_input_name] = _create_channel_with_empty_artifact(
          data_type)
      continue

    input_channel_parameters[python_input_name] = _create_channel_parameter(
        input_spec.type)
    if input_spec.optional:
      default_input_channels[python_input_name] = (
          _create_channel_with_empty_artifact(input_spec.type))

  for output_spec in component_spec.outputs or []:
    python_output_name = output_name_to_python[output_spec.name]
    output_channel_parameters[python_output_name] = _create_channel_parameter(
        output_spec.type)
    output_channels[python_output_name] = _create_channel_with_empty_artifact(
        output_spec.type)

  component_name = component_spec.name or 'Component'
  component_class_name = _sanitize_python_class_name(component_name)
  component_class_doc = (component_spec.name or '') + '\n'
  component_class_doc += (component_spec.description or '')

  tfx_component_spec_class = type(
      component_class_name + 'Spec',
      (tfx_component_spec.ComponentSpec,),
      dict(
          PARAMETERS=execution_parameters,
          INPUTS=input_channel_parameters,
          OUTPUTS=output_channel_parameters,
          __doc__=component_class_doc,
      ),
  )

  component_arguments = {}
  for input_spec in component_spec.inputs or []:
    pythonic_input_name = input_name_to_python[input_spec.name]
    if input_spec.name in inputs_that_are_input_artifact_uris:
      cmd_argument = '{{{{input_dict["{name}"][0].uri}}}}'.format(
          name=pythonic_input_name)
    elif input_spec.name in inputs_that_are_output_artifact_uris:
      cmd_argument = '{{{{output_dict["{name}"][0].uri}}}}'.format(
          name=pythonic_input_name)
    elif input_spec.name in inputs_that_are_execution_properties:
      cmd_argument = '{{{{exec_properties["{name}"]}}}}'.format(
          name=pythonic_input_name)
    else:
      # The {{input_dict["name"][0].value}} placeholder is not implemented yet.
      # @jxzheng and @avolkov are working on this.
      cmd_argument = '{{{{input_dict["{name}"][0].value}}}}'.format(
          name=pythonic_input_name)
    component_arguments[input_spec.name] = cmd_argument

  # TODO(avolkov): Remove these generators once cl/294536017 is submitted
  def input_path_uri_generator(name):
    return '{{{{input_dict["{name}"][0].uri}}}}'.format(
        name=input_name_to_python[name])

  def output_path_uri_generator(name):
    return '{{{{output_dict["{name}"][0].uri}}}}'.format(
        name=output_name_to_python[name])

  resolved_cmd = components._components._resolve_command_line_and_paths(  # pylint: disable=protected-access
      component_spec=component_spec,
      arguments=component_arguments,
      # TODO(avolkov): Remove this workaround when cl/294536017 is submitted
      input_path_generator=input_path_uri_generator,
      output_path_generator=output_path_uri_generator,
  )

  resolved_command = resolved_cmd.command
  resolved_args = resolved_cmd.args

  def tfx_component_class_init(self, **kwargs):
    instance_name = kwargs.pop('instance_name', None)
    arguments = {}
    arguments.update(default_input_channels)
    arguments.update(output_channels)
    arguments.update(kwargs)

    BaseComponent.__init__(
        self,
        spec=self.__class__.SPEC_CLASS(**arguments),
        instance_name=instance_name,
    )

  tfx_component_class = type(
      component_class_name,
      (BaseComponent,),
      dict(
          SPEC_CLASS=tfx_component_spec_class,
          EXECUTOR_SPEC=ExecutorContainerSpec(
              image=container_spec.image,
              command=resolved_command,
              args=resolved_args,
          ),
          __init__=tfx_component_class_init,
          __doc__=component_class_doc,
      ),
  )

  return tfx_component_class


def _create_tfx_task_from_component_spec_and_arguments(
    component_spec: structures.ComponentSpec,
    arguments: Mapping[str, Any],
    component_ref: structures.ComponentReference,
) -> BaseComponent:
  del component_ref
  tfx_component_class = create_tfx_component_class_from_spec(component_spec)
  tfx_task = tfx_component_class(**arguments)
  return tfx_task


class enable_new_components(contextlib.ContextDecorator):  # pylint: disable=invalid-name

  def __enter__(self):
    self.old_handler = components._components._container_task_constructor  # pylint: disable=protected-access
    components._components._container_task_constructor = _create_tfx_task_from_component_spec_and_arguments  # pylint: disable=protected-access

  def __exit__(self, exc_type, exc, exc_tb):
    components._components._container_task_constructor = self.old_handler  # pylint: disable=protected-access
