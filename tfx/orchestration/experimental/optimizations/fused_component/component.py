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

from typing import Optional, Text, List

from tfx.components.base import base_component
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.components.statistics_gen import executor
from tfx.types.standard_component_specs import FusedComponentSpec


class FusedComponent(base_component.BaseComponent):
  """Official TFX fused component.

  The FusedComponent acts as a wrapper for Apache Beam based components whose
  reads,execution, and writes can be fused to prevent extra file I/O.

  """

  SPEC_CLASS = FusedComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               subgraph: List[base_node.BaseNode],
               instance_name: Optional[Text] = None):
    """Construct a FusedComponent.

    Args:
      components: logical components of this FusedComponent, in topological
        order.
      instance_name: Optional name assigned to this specific instance of
        FusedComponent.  Required only if multiple FusedComponents are declared
        in the same pipeline.
    """
    # Need to change FusedComponentSpec dynamically for each FusedComponent()
    # instance. Create a subclass, modify INPUTS, OUTPUTS, etc.
    spec = FusedComponentSpec()
    super(FusedComponent, self).__init__(spec=spec,
                                         instance_name=instance_name)
    self.subgraph = subgraph

  def in_subgraph(self, component: base_node.BaseNode):
    return component in self.subgraph

  def get_subgraph(self):
    return self.subgraph
