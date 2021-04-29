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
"""Tests for using parquet_executor with example_gen component."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from unittest import mock
import tensorflow as tf
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.proto import example_gen_pb2

from ml_metadata.proto import metadata_store_pb2


class ExampleGenComponentWithParquetExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExampleGenComponentWithParquetExecutorTest, self).setUp()
    # Create input_base.
    input_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata')
    self.parquet_dir_path = os.path.join(input_data_dir, 'external')

    # Create input_config.
    self.input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='parquet',
                                    pattern='parquet/*.parquet'),
    ])

    # Create output_config.
    self.output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))

  @mock.patch.object(publisher, 'Publisher')
  def testRun(self, mock_publisher):
    mock_publisher.return_value.publish_execution.return_value = {}

    example_gen = FileBasedExampleGen(
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            parquet_executor.Executor),
        input_base=self.parquet_dir_path,
        input_config=self.input_config,
        output_config=self.output_config).with_id('ParquetExampleGen')

    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    pipeline_root = os.path.join(output_data_dir, 'Test')
    fileio.makedirs(pipeline_root)
    pipeline_info = data_types.PipelineInfo(
        pipeline_name='Test', pipeline_root=pipeline_root, run_id='123')

    driver_args = data_types.DriverArgs(enable_cache=True)

    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    metadata_connection = metadata.Metadata(connection_config)

    launcher = in_process_component_launcher.InProcessComponentLauncher.create(
        component=example_gen,
        pipeline_info=pipeline_info,
        driver_args=driver_args,
        metadata_connection=metadata_connection,
        beam_pipeline_args=[],
        additional_pipeline_args={})
    self.assertEqual(
        launcher._component_info.component_type,
        '.'.join([FileBasedExampleGen.__module__,
                  FileBasedExampleGen.__name__]))

    launcher.launch()
    mock_publisher.return_value.publish_execution.assert_called_once()

    # Check output paths.
    self.assertTrue(fileio.exists(os.path.join(pipeline_root, example_gen.id)))


if __name__ == '__main__':
  tf.test.main()
