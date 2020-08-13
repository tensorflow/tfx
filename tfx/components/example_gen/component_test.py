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
"""Tests for tfx.components.example_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import any_pb2
from google.protobuf import json_format
from tfx.components.base import base_driver
from tfx.components.base import executor_spec
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen import component
from tfx.components.example_gen import driver
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts


class TestExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):

  def GetInputSourceToExamplePTransform(self):
    pass


class TestQueryBasedExampleGenComponent(component.QueryBasedExampleGen):

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(TestExampleGenExecutor)

  def __init__(self,
               input_config,
               output_config=None,
               example_artifacts=None,
               instance_name=None):
    super(TestQueryBasedExampleGenComponent, self).__init__(
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        instance_name=instance_name)


class TestFileBasedExampleGenComponent(component.FileBasedExampleGen):

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(TestExampleGenExecutor)

  def __init__(self,
               input_base,
               input_config=None,
               output_config=None,
               example_artifacts=None,
               instance_name=None):
    super(TestFileBasedExampleGenComponent, self).__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        example_artifacts=example_artifacts,
        instance_name=instance_name)


class ComponentTest(tf.test.TestCase):

  def testConstructSubclassQueryBased(self):
    example_gen = TestQueryBasedExampleGenComponent(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='query'),
        ]))
    self.assertEqual({}, example_gen.inputs.get_all())
    self.assertEqual(base_driver.BaseDriver, example_gen.driver_class)
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     example_gen.outputs['examples'].type_name)
    self.assertIsNone(example_gen.exec_properties.get('custom_config'))
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))

  def testConstructSubclassFileBased(self):
    example_gen = TestFileBasedExampleGenComponent(input_base='path')
    self.assertIn('input_base', example_gen.exec_properties)
    self.assertEqual(driver.Driver, example_gen.driver_class)
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     example_gen.outputs['examples'].type_name)
    self.assertIsNone(example_gen.exec_properties.get('custom_config'))
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))

  def testConstructCustomExecutor(self):
    example_gen = component.FileBasedExampleGen(
        input_base='path',
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            TestExampleGenExecutor))
    self.assertEqual(driver.Driver, example_gen.driver_class)
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))

  def testConstructWithOutputConfig(self):
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
        ]))
    example_gen = TestFileBasedExampleGenComponent(
        input_base='path', output_config=output_config)
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))

    stored_output_config = example_gen_pb2.Output()
    json_format.Parse(example_gen.exec_properties['output_config'],
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
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))

    stored_input_config = example_gen_pb2.Input()
    json_format.Parse(example_gen.exec_properties['input_config'],
                      stored_input_config)
    self.assertEqual(input_config, stored_input_config)

  def testConstructWithCustomConfig(self):
    custom_config = example_gen_pb2.CustomConfig(custom_config=any_pb2.Any())
    example_gen = component.FileBasedExampleGen(
        input_base='path',
        custom_config=custom_config,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            TestExampleGenExecutor))

    stored_custom_config = example_gen_pb2.CustomConfig()
    json_format.Parse(example_gen.exec_properties['custom_config'],
                      stored_custom_config)
    self.assertEqual(custom_config, stored_custom_config)


if __name__ == '__main__':
  tf.test.main()
