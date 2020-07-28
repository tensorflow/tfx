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
"""Stub component launcher for launching stub executors in KFP."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.experimental.pipeline_testing import base_stub_executor
from tfx.experimental.pipeline_testing import stub_component_launcher
from tfx.experimental.templates.taxi.pipeline import configs

class StubComponentLauncher(stub_component_launcher.StubComponentLauncher):
  """Responsible for launching stub executors in KFP Template.
  This stub component launcher cannot be defined in the kubeflow_dag_runner.py
  because launcher class is imported by the module path.
  """
  def __init__(self, **kwargs):
    super(StubComponentLauncher, self).__init__(**kwargs)

    # TODO(StubExecutor): GCS directory where KFP outputs are recorded
    self.test_data_dir = "gs://{}/testdata".format(configs.GCS_BUCKET_NAME)
    # TODO(StubExecutor): customize self.stubbed_component_ids to replace components
    # with BaseStubExecutor
    self.stubbed_component_ids = ['CsvExampleGen', 'StatisticsGen',
                                  'SchemaGen', 'ExampleValidator',
                                  'Trainer', 'Transform', 'Evaluator', 'Pusher']
    # TODO(StubExecutor): (Optional) Use stubbed_component_map to insert custom stub
    # executor class as a value and component id as a key.
    self.stubbed_component_map = {}
    for c_id in self.stubbed_component_ids:
      self.stubbed_component_map[c_id] = base_stub_executor.BaseStubExecutor

def get_stub_launcher_class(stub_launcher: Type[StubComponentLauncher],
      test_data_dir: Text, 
      stubbed_component_ids: List[Text],
      stubbed_component_map: Dict[Text, Type[base_stub_executor.BaseStubExecutor]]
  ) -> Type[StubComponentLauncher]:
  """Returns a StubComponentLauncher class.
  Returns:
    StubComponentLauncher class holding stub executors.
  """
  stub_launcher.stubbed_component_map = dict(stubbed_component_map)
  for component_id in stubbed_component_ids:
    stub_launcher.stubbed_component_map[component_id] = \
                    base_stub_executor.BaseStubExecutor
  stub_launcher.test_data_dir = test_data_dir
  return stub_launcher
