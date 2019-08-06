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
"""Tests for tfx.components.setup.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tfx.components.setup import component
from tfx.orchestration import data_types


class SetupTest(tf.test.TestCase):

  def test_construct(self):
    setup_componet = component.Setup()
    self.assertEqual(component.SetupExecutor, setup_componet.executor_class)
    self.assertEqual(component.SetupDriver, setup_componet.driver_class)

  def test_driver(self):
    mock_metadata = tf.test.mock.Mock()

    driver_args = data_types.DriverArgs(enable_cache=True)
    pipeline_info = data_types.PipelineInfo(
        pipeline_name='my_pipeline_name',
        pipeline_root=os.environ.get('TEST_TMP_DIR', self.get_temp_dir()),
        run_id='my_run_id')
    component_info = data_types.ComponentInfo(
        component_type='setup', component_id='my_component_id')

    execution_id = 1
    mock_metadata.register_execution.side_effect = [execution_id]

    driver = component.SetupDriver(metadata_handler=mock_metadata)
    execution_decision = driver.pre_execution(
        input_dict={},
        output_dict={},
        exec_properties={},
        driver_args=driver_args,
        pipeline_info=pipeline_info,
        component_info=component_info)
    self.assertFalse(execution_decision.use_cached_results)
    self.assertEqual(execution_decision.execution_id, 1)


if __name__ == '__main__':
  tf.test.main()
