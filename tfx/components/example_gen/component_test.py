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
from tfx.components.base import base_driver
from tfx.components.example_gen import base_example_gen_executor
from tfx.components.example_gen import component
from tfx.components.example_gen import driver
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types


class TestExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):

  def GetInputSourceToExamplePTransform(self):
    pass


class TestExampleGenComponent(component._ExampleGen):

  EXECUTOR_CLASS = TestExampleGenExecutor

  def __init__(self,
               input_config=None,
               output_config=None,
               name=None,
               example_artifacts=None):
    super(TestExampleGenComponent, self).__init__(
        input_config=input_config,
        output_config=output_config,
        component_name='TestExampleGenComponent',
        example_artifacts=example_artifacts,
        name=name)


class TestFileBasedExampleGenComponent(component._FileBasedExampleGen):

  EXECUTOR_CLASS = TestExampleGenExecutor

  def __init__(self,
               input_base,
               input_config=None,
               output_config=None,
               name=None,
               example_artifacts=None):
    super(TestFileBasedExampleGenComponent, self).__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        component_name='TestFileBasedExampleGenComponent',
        example_artifacts=example_artifacts,
        name=name)


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    example_gen = component._ExampleGen(input_config=None)
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_file_based(self):
    input_base = types.TfxArtifact(type_name='ExternalPath')
    example_gen = component._FileBasedExampleGen(
        input_base=channel.as_channel([input_base]))
    self.assertEqual(driver.Driver, example_gen.driver_class)
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_subclass(self):
    example_gen = TestExampleGenComponent()
    self.assertEqual(base_driver.BaseDriver, example_gen.driver_class)
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_subclass_file_based(self):
    input_base = types.TfxArtifact(type_name='ExternalPath')
    example_gen = TestFileBasedExampleGenComponent(
        input_base=channel.as_channel([input_base]))
    self.assertIn('input_base', example_gen.inputs.get_all())
    self.assertEqual(driver.Driver, example_gen.driver_class)
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_with_output_config(self):
    input_base = types.TfxArtifact(type_name='ExternalPath')
    example_gen = component._FileBasedExampleGen(
        input_base=channel.as_channel([input_base]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
                example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
            ])))
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)
    self.assertEqual('test', artifact_collection[2].split)

  def test_construct_with_input_config(self):
    input_base = types.TfxArtifact(type_name='ExternalPath')
    example_gen = component._FileBasedExampleGen(
        input_base=channel.as_channel([input_base]),
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*'),
            example_gen_pb2.Input.Split(name='test', pattern='test/*')
        ]))
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)
    self.assertEqual('test', artifact_collection[2].split)

  def test_construct_without_input_base(self):
    example_gen = component._ExampleGen(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='query'),
        ]))
    self.assertEqual({}, example_gen.inputs.get_all())
    self.assertEqual(base_driver.BaseDriver, example_gen.driver_class)
    self.assertEqual('ExamplesPath', example_gen.outputs['examples'].type_name)
    artifact_collection = example_gen.outputs['examples'].get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)


if __name__ == '__main__':
  tf.test.main()
