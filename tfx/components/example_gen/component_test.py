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
"""Tests for tfx.components.example_gen.component."""

import tensorflow as tf
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen import component
from tfx.components.example_gen import driver
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import proto_utils

from google.protobuf import any_pb2


class TestExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):

  def GetInputSourceToExamplePTransform(self):
    pass


class TestQueryBasedExampleGenComponent(component.QueryBasedExampleGen):

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(TestExampleGenExecutor)

  def __init__(self,
               input_config,
               output_config=None,
               output_data_format=example_gen_pb2.FORMAT_TF_EXAMPLE,
               output_file_format=example_gen_pb2.FORMAT_TFRECORDS_GZIP,
               ):
    super().__init__(
        input_config=input_config,
        output_config=output_config,
        output_data_format=output_data_format,
        output_file_format=output_file_format,
    )


class TestFileBasedExampleGenComponent(component.FileBasedExampleGen):

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(TestExampleGenExecutor)

  def __init__(self, input_base, input_config=None, output_config=None):
    super().__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config)


class ComponentTest(tf.test.TestCase):

  def testConstructSubclassQueryBased(self):
    example_gen = TestQueryBasedExampleGenComponent(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='query'),
        ]))
    self.assertEqual({}, example_gen.inputs)
    self.assertEqual(driver.QueryBasedDriver, example_gen.driver_class)
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME,
        example_gen.outputs[standard_component_specs.EXAMPLES_KEY].type_name)
    self.assertEqual(
        example_gen.exec_properties[
            standard_component_specs.OUTPUT_DATA_FORMAT_KEY],
        example_gen_pb2.FORMAT_TF_EXAMPLE)
    self.assertEqual(
        example_gen.exec_properties[
            standard_component_specs.OUTPUT_FILE_FORMAT_KEY],
        example_gen_pb2.FORMAT_TFRECORDS_GZIP)
    self.assertIsNone(
        example_gen.exec_properties.get(
            standard_component_specs.CUSTOM_CONFIG_KEY))

  def testConstructSubclassQueryBasedWithInvalidOutputDataFormat(self):
    self.assertRaises(
        ValueError,
        TestQueryBasedExampleGenComponent,
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='query'),
        ]),
        output_data_format=-1  # not exists
    )

  def testConstructSubclassQueryBasedWithInvalidOutputFileFormat(self):
    self.assertRaises(
        ValueError,
        TestQueryBasedExampleGenComponent,
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='query'),
        ]),
        output_file_format=-1  # not exists
    )

  def testConstructSubclassFileBased(self):
    example_gen = TestFileBasedExampleGenComponent(input_base='path')
    self.assertIn(standard_component_specs.INPUT_BASE_KEY,
                  example_gen.exec_properties)
    self.assertEqual(driver.FileBasedDriver, example_gen.driver_class)
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME,
        example_gen.outputs[standard_component_specs.EXAMPLES_KEY].type_name)
    self.assertIsNone(
        example_gen.exec_properties.get(
            standard_component_specs.CUSTOM_CONFIG_KEY))

  def testConstructCustomExecutor(self):
    example_gen = component.FileBasedExampleGen(
        input_base='path',
        custom_executor_spec=executor_spec.BeamExecutorSpec(
            TestExampleGenExecutor))
    self.assertEqual(driver.FileBasedDriver, example_gen.driver_class)
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME,
        example_gen.outputs[standard_component_specs.EXAMPLES_KEY].type_name)

  def testConstructWithOutputConfig(self):
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
        ]))
    example_gen = TestFileBasedExampleGenComponent(
        input_base='path', output_config=output_config)
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME,
        example_gen.outputs[standard_component_specs.EXAMPLES_KEY].type_name)

    stored_output_config = example_gen_pb2.Output()
    proto_utils.json_to_proto(
        example_gen.exec_properties[standard_component_specs.OUTPUT_CONFIG_KEY],
        stored_output_config)
    self.assertEqual(output_config, stored_output_config)

  def testConstructWithInputConfig(self):
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train/*'),
        example_gen_pb2.Input.Split(name='eval', pattern='eval/*'),
        example_gen_pb2.Input.Split(name='test', pattern='test/*')
    ])
    example_gen = TestFileBasedExampleGenComponent(
        input_base='path', input_config=input_config)
    self.assertEqual(
        standard_artifacts.Examples.TYPE_NAME,
        example_gen.outputs[standard_component_specs.EXAMPLES_KEY].type_name)

    stored_input_config = example_gen_pb2.Input()
    proto_utils.json_to_proto(
        example_gen.exec_properties[standard_component_specs.INPUT_CONFIG_KEY],
        stored_input_config)
    self.assertEqual(input_config, stored_input_config)

  def testConstructWithCustomConfig(self):
    custom_config = example_gen_pb2.CustomConfig(custom_config=any_pb2.Any())
    example_gen = component.FileBasedExampleGen(
        input_base='path',
        custom_config=custom_config,
        custom_executor_spec=executor_spec.BeamExecutorSpec(
            TestExampleGenExecutor))

    stored_custom_config = example_gen_pb2.CustomConfig()
    proto_utils.json_to_proto(
        example_gen.exec_properties[standard_component_specs.CUSTOM_CONFIG_KEY],
        stored_custom_config)
    self.assertEqual(custom_config, stored_custom_config)

  def testConstructWithStaticRangeConfig(self):
    range_config = range_config_pb2.RangeConfig(
        static_range=range_config_pb2.StaticRange(
            start_span_number=1, end_span_number=1))
    example_gen = component.FileBasedExampleGen(
        input_base='path',
        range_config=range_config,
        custom_executor_spec=executor_spec.BeamExecutorSpec(
            TestExampleGenExecutor))
    stored_range_config = range_config_pb2.RangeConfig()
    proto_utils.json_to_proto(
        example_gen.exec_properties[standard_component_specs.RANGE_CONFIG_KEY],
        stored_range_config)
    self.assertEqual(range_config, stored_range_config)


if __name__ == '__main__':
  tf.test.main()
