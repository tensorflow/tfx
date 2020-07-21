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

from tfx.experimental.pipeline_testing import base_stub_executor
from tfx.experimental.pipeline_testing import stub_component_launcher

class TemplateStubComponentLauncher(
    stub_component_launcher.StubComponentLauncher):
  """Responsible for launching stub executors in KFP Template."""

  def __init__(self, **kwargs):
    super(TemplateStubComponentLauncher, self).__init__(**kwargs)

    self.test_data_dir = '' # TODO: GCS directory where KFP outputs are recorded
    # TODO: (Optional) insert custom stub executors in self.stubbed_component_map with
    # component id as a key and custom stub executor class as value.
    self.stubbed_component_map = {}
    # TODO: customize self,stubbed_component_ids to replace component with BaseStubExecutor
    self.stubbed_component_ids = ['CsvExampleGen', 'StatisticsGen',
                                  'SchemaGen', 'ExampleValidator',
                                  'Trainer', 'Transform', 'Evaluator', 'Pusher']
    for c_id in self.stubbed_component_ids:
      self.stubbed_component_map[c_id] = base_stub_executor.BaseStubExecutor
