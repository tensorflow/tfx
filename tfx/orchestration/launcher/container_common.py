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

import copy

from typing import Any, Dict, List, Text

import jinja2

from tfx import types
from tfx.components.base import executor_spec


def resolve_container_template(
    container_spec_tmpl: executor_spec.ExecutorContainerSpec,
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
  # Making a copy to keep any additional attributes
  resolved_spec = copy.copy(container_spec_tmpl)
  resolved_spec.image = _render_text(container_spec_tmpl.image, context)
  resolved_spec.command = _render_items(container_spec_tmpl.command, context)
  resolved_spec.args = _render_items(container_spec_tmpl.args, context)
  resolved_spec.input_path_uris = {
      path: _render_text(uri, context)
      for path, uri in (container_spec_tmpl.input_path_uris or {}).items()
  }
  resolved_spec.output_path_uris = {
      path: _render_text(uri, context)
      for path, uri in (container_spec_tmpl.output_path_uris or {}).items()
  }
  return resolved_spec


def _render_items(items: List[Text], context: Dict[Text, Any]) -> List[Text]:
  if not items:
    return items

  return [_render_text(item, context) for item in items]


def _render_text(text: Text, context: Dict[Text, Any]) -> Text:
  return jinja2.Template(text).render(context)


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
