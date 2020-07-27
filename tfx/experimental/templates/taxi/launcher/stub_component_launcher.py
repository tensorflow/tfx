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
"""Template stub component launcher for launching stub executors in KFP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, List, Text, Type

from tfx.experimental.pipeline_testing import base_stub_executor
from tfx.experimental.pipeline_testing import stub_component_launcher

class StubComponentLauncher(stub_component_launcher.StubComponentLauncher):
  """Responsible for launching stub executors in KFP Template."""

  def __init__(self, **kwargs):
    super(TemplateStubComponentLauncher, self).__init__(**kwargs)

def get_stub_launcher_class(
    test_data_dir: Text, stubbed_component_ids: List[Text],
    stubbed_component_map: Dict[Text, Type[base_stub_executor.BaseStubExecutor]]
) -> Type[StubComponentLauncher]:
  """Returns a StubComponentLauncher class.

  Args:
    test_data_dir: GCS path where pipeline outputs are recorded.
    stubbed_component_ids: List of component ids that should be replaced with a
      BaseStubExecutor.
    stubbed_component_map: Dictionary holding user-defined stub executor. These
      user-defined stub executors must inherit from
      base_stub_executor.BaseStubExecutor.

  Returns:
    StubComponentLauncher class holding stub executors.
  """
  cls = StubComponentLauncher
  cls.stubbed_component_map = dict(stubbed_component_map)
  for component_id in stubbed_component_ids:
    cls.stubbed_component_map[component_id] = \
                    base_stub_executor.BaseStubExecutor
  cls.test_data_dir = test_data_dir
  return cls
