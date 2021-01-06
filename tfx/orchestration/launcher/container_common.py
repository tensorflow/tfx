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
"""Common code shared by container based launchers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Optional, Text, Union

# TODO(b/176812386): Deprecate usage of jinja2 for placeholders.
import jinja2

from tfx import types
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import executor_spec


def resolve_container_template(
    container_spec_tmpl: Union[executor_spec.ExecutorContainerSpec,
                               executor_specs.TemplatedExecutorContainerSpec],
    input_dict: Dict[Text, List[types.Artifact]],
    output_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any]) -> executor_spec.ExecutorContainerSpec:
  """Resolves Jinja2 template languages from an executor container spec.

  Args:
    container_spec_tmpl: the container spec template to be resolved.
    input_dict: Dictionary of input artifacts consumed by this component.
    output_dict: Dictionary of output artifacts produced by this component.
    exec_properties: Dictionary of execution properties.

  Returns:
    A resolved container spec.
  """
  context = {
      'input_dict': input_dict,
      'output_dict': output_dict,
      'exec_properties': exec_properties,
  }
  if isinstance(container_spec_tmpl,
                executor_specs.TemplatedExecutorContainerSpec):
    return executor_spec.ExecutorContainerSpec(
        image=container_spec_tmpl.image,
        command=_resolve_container_command_line(
            cmd_args=container_spec_tmpl.command,
            input_dict=input_dict,
            output_dict=output_dict,
            exec_properties=exec_properties,
        ),
    )
  return executor_spec.ExecutorContainerSpec(
      image=_render_text(container_spec_tmpl.image, context),
      command=_render_items(container_spec_tmpl.command, context),
      args=_render_items(container_spec_tmpl.args, context))


def _render_items(items: List[Text], context: Dict[Text, Any]) -> List[Text]:
  if not items:
    return items

  return [_render_text(item, context) for item in items]


def _render_text(text: Text, context: Dict[Text, Any]) -> Text:
  return jinja2.Template(text).render(context)


def _resolve_container_command_line(
    cmd_args: Optional[List[
        placeholders.CommandlineArgumentType]],
    input_dict: Dict[Text, List[types.Artifact]],
    output_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],
) -> List[Text]:
  """Resolves placeholders in the command line of a container.

  Args:
    cmd_args: command line args to resolve.
    input_dict: Dictionary of input artifacts consumed by this component.
    output_dict: Dictionary of output artifacts produced by this component.
    exec_properties: Dictionary of execution properties.

  Returns:
    Resolved command line.
  """

  def expand_command_line_arg(
      cmd_arg: placeholders.CommandlineArgumentType,
  ) -> Text:
    """Resolves a single argument."""
    if isinstance(cmd_arg, str):
      return cmd_arg
    elif isinstance(cmd_arg, placeholders.InputValuePlaceholder):
      if cmd_arg.input_name in exec_properties:
        return str(exec_properties[cmd_arg.input_name])
      else:
        artifact = input_dict[cmd_arg.input_name][0]
        return str(artifact.value)
    elif isinstance(cmd_arg, placeholders.InputUriPlaceholder):
      return input_dict[cmd_arg.input_name][0].uri
    elif isinstance(cmd_arg, placeholders.OutputUriPlaceholder):
      return output_dict[cmd_arg.output_name][0].uri
    elif isinstance(cmd_arg, placeholders.ConcatPlaceholder):
      resolved_items = [expand_command_line_arg(item) for item in cmd_arg.items]
      for item in resolved_items:
        if not isinstance(item, (str, Text)):
          raise TypeError('Expanded item "{}" has incorrect type "{}"'.format(
              item, type(item)))
      return ''.join(resolved_items)
    else:
      raise TypeError(
          ('Unsupported type of command-line arguments: "{}".'
           ' Supported types are {}.')
          .format(type(cmd_arg), str(executor_specs.CommandlineArgumentType)))

  resolved_command_line = []
  for cmd_arg in (cmd_args or []):
    resolved_cmd_arg = expand_command_line_arg(cmd_arg)
    if not isinstance(resolved_cmd_arg, (str, Text)):
      raise TypeError(
          'Resolved argument "{}" (type="{}") is not a string.'.format(
              resolved_cmd_arg, type(resolved_cmd_arg)))
    resolved_command_line.append(resolved_cmd_arg)

  return resolved_command_line


def to_swagger_dict(config: Any) -> Any:
  """Converts a config object to a swagger API dict.

  This utility method recursively converts swagger code generated configs into
  a valid swagger dictionary. This method is trying to workaround a bug
  (https://github.com/swagger-api/swagger-codegen/issues/8948)
  from swagger generated code

  Args:
    config: The config object. It can be one of List, Dict or a Swagger code
      generated object, which has a `attribute_map` attribute.

  Returns:
    The original object with all Swagger generated object replaced with
    dictionary object.
  """
  if isinstance(config, list):
    return [to_swagger_dict(x) for x in config]
  if hasattr(config, 'attribute_map'):
    return {
        swagger_name: to_swagger_dict(getattr(config, key))
        for (key, swagger_name) in config.attribute_map.items()
        if getattr(config, key)
    }
  if isinstance(config, dict):
    return {key: to_swagger_dict(value) for key, value in config.items()}
  return config
