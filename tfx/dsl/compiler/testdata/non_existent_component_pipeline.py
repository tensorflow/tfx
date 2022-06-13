# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Sample pipeline referencing an non existent node."""

import os
from typing import List

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.components.base import base_node
from tfx.orchestration import pipeline

_pipeline_name = 'non_existent_node'
_pipeline_root = os.path.join('pipeline', _pipeline_name)


@component()
def UpstreamComponent():  # pylint: disable=invalid-name
  pass


@component()
def NonExistentComponent():  # pylint: disable=invalid-name
  pass


def generate_components() -> List[base_node.BaseNode]:  # pylint: disable=invalid-name
  upstream_component = UpstreamComponent()
  upstream_component.add_downstream_node(NonExistentComponent())
  return [upstream_component]


def create_test_pipeline():  # pylint: disable=invalid-name
  return pipeline.Pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      components=generate_components())
