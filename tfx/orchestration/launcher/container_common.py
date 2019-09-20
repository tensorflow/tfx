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

import jinja2

from typing import Any, Dict, List, Text

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
