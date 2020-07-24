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
"""Tests for tfx.components.base.base_component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.proto import example_gen_pb2
from tfx.types import component_spec
from tfx.utils import json_utils


class _InputArtifact(types.Artifact):
  TYPE_NAME = "InputArtifact"


class _OutputArtifact(types.Artifact):
  TYPE_NAME = "OutputArtifact"


class _BasicComponentSpec(types.ComponentSpec):

  PARAMETERS = {
      "folds":
          component_spec.ExecutionParameter(type=int),
      "proto":
          component_spec.ExecutionParameter(
              type=example_gen_pb2.Input, optional=True),
  }
  INPUTS = {
      "input": component_spec.ChannelParameter(type=_InputArtifact),
  }
  OUTPUTS = {
      "output": component_spec.ChannelParameter(type=_OutputArtifact),
  }


class _BasicComponent(base_component.BaseComponent):

  SPEC_CLASS = _BasicComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self,
               spec: types.ComponentSpec = None,
               folds: int = None,
               input: types.Channel = None):  # pylint: disable=redefined-builtin
    if not spec:
      output = types.Channel(type=_OutputArtifact)
      spec = _BasicComponentSpec(folds=folds, input=input, output=output)
    super(_BasicComponent, self).__init__(spec=spec)


class ComponentTest(tf.test.TestCase):

  def testComponentBasic(self):
    input_channel = types.Channel(type=_InputArtifact)
    component = _BasicComponent(folds=10, input=input_channel)
    self.assertEqual(component.id, "_BasicComponent")
    self.assertIs(input_channel, component.inputs["input"])
    self.assertIsInstance(component.outputs["output"], types.Channel)
    self.assertEqual(component.outputs["output"].type, _OutputArtifact)
    self.assertEqual(component.outputs["output"].type_name, "OutputArtifact")

  def testComponentSpecType(self):

    with self.assertRaisesRegexp(
        ValueError,
        'expects "spec" argument to be an instance of types.ComponentSpec'):
      _ = _BasicComponent(spec=object())  # pytype: disable=wrong-arg-types

  def testComponentSpecClass(self):

    class MissingSpecComponent(base_component.BaseComponent):

      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      MissingSpecComponent(spec=object())  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegexp(
        TypeError, "expects SPEC_CLASS property to be a subclass of "
        "types.ComponentSpec"):
      MissingSpecComponent._validate_component_class()

    class InvalidSpecComponent(base_component.BaseComponent):

      SPEC_CLASSES = object()
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)

    with self.assertRaisesRegexp(
        TypeError, "expects SPEC_CLASS property to be a subclass of "
        "types.ComponentSpec"):
      InvalidSpecComponent._validate_component_class()

  def testComponentExecutorClass(self):

    class MissingExecutorComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      MissingExecutorComponent(spec=object())  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegexp(
        TypeError, "expects EXECUTOR_SPEC property to be an instance of "
        "ExecutorSpec"):
      MissingExecutorComponent._validate_component_class()

    class InvalidExecutorComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_SPEC = object()

    with self.assertRaisesRegexp(
        TypeError, "expects EXECUTOR_SPEC property to be an instance of "
        "ExecutorSpec"):
      InvalidExecutorComponent._validate_component_class()

  def testComponentCustomExecutor(self):

    class EmptyComponentSpec(types.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class MyComponent(base_component.BaseComponent):

      SPEC_CLASS = EmptyComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)

    class MyCustomExecutor(base_executor.BaseExecutor):
      pass

    custom_executor_component = MyComponent(
        spec=EmptyComponentSpec(),
        custom_executor_spec=executor_spec.ExecutorClassSpec(MyCustomExecutor))
    self.assertEqual(custom_executor_component.executor_spec.executor_class,
                     MyCustomExecutor)

    with self.assertRaisesRegexp(TypeError,
                                 "should be an instance of ExecutorSpec"):
      MyComponent(spec=EmptyComponentSpec(), custom_executor_spec=object)

  def testComponentDriverClass(self):

    class InvalidDriverComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)
      DRIVER_CLASS = object()

    with self.assertRaisesRegexp(
        TypeError, "expects DRIVER_CLASS property to be a subclass of "
        "base_driver.BaseDriver"):
      InvalidDriverComponent._validate_component_class()

  def testJsonify(self):
    input_channel = types.Channel(
        type=_InputArtifact, artifacts=[_InputArtifact()])
    component = _BasicComponent(folds=10, input=input_channel)
    json_dict = json_utils.dumps(component)
    recovered_component = json_utils.loads(json_dict)
    self.assertEqual(recovered_component.__class__, component.__class__)
    self.assertEqual(recovered_component.component_id, "_BasicComponent")
    self.assertEqual(input_channel.type,
                     recovered_component.inputs["input"].type)
    self.assertEqual(len(recovered_component.inputs["input"].get()), 1)
    self.assertIsInstance(recovered_component.outputs["output"], types.Channel)
    self.assertEqual(recovered_component.outputs["output"].type,
                     _OutputArtifact)
    self.assertEqual(recovered_component.outputs["output"].type_name,
                     "OutputArtifact")
    self.assertEqual(recovered_component.driver_class, component.driver_class)
    # Test re-dump.
    new_json_dict = json_utils.dumps(recovered_component)
    self.assertEqual(new_json_dict, json_dict)

  def testGetId(self):
    self.assertEqual(_BasicComponent.get_id(), "_BasicComponent")
    self.assertEqual(
        _BasicComponent.get_id(instance_name="my_instance"),
        "_BasicComponent.my_instance")

  def testTaskDependency(self):
    channel_1 = types.Channel(type=_InputArtifact)
    component_1 = _BasicComponent(folds=10, input=channel_1)
    channel_2 = types.Channel(type=_InputArtifact)
    component_2 = _BasicComponent(folds=10, input=channel_2)
    self.assertEqual(False, component_2 in component_1.downstream_nodes)
    self.assertEqual(False, component_1 in component_2.upstream_nodes)
    component_1.add_downstream_node(component_2)
    self.assertEqual(True, component_2 in component_1.downstream_nodes)
    self.assertEqual(True, component_1 in component_2.upstream_nodes)

if __name__ == "__main__":
  tf.test.main()
