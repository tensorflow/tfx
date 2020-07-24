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
"""TFX IPython formatter integration.

Note: these APIs are **experimental** and major changes to interface and
functionality are expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

from tfx.components.base import base_node


class ExecutionResult(object):
  """Execution result from a component launch."""

  def __init__(self, component: base_node.BaseNode, execution_id: int):
    self.component = component
    self.execution_id = execution_id

  def __repr__(self):
    outputs_parts = []
    for name, chan in self.component.outputs.items():
      repr_string = '%s: %s' % (name, repr(chan))
      for line in repr_string.split('\n'):
        outputs_parts.append(line)
    outputs_str = '\n'.join('        %s' % line for line in outputs_parts)
    return ('ExecutionResult(\n    component_id: %s'
            '\n    execution_id: %s'
            '\n    outputs:\n%s'
            ')') % (self.component.id,
                    self.execution_id,
                    outputs_str)
