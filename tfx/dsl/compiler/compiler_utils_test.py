# Lint as: python2, python3
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
"""Tests for tfx.dsl.compiler.compiler_utils."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow as tf
from ml_metadata.proto import metadata_store_pb2
from tfx import types
from tfx.components import CsvExampleGen
from tfx.components import ResolverNode
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils.dsl_utils import external_input


class EmptyComponentSpec(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {}


class EmptyComponent(base_component.BaseComponent):

  SPEC_CLASS = EmptyComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, name):
    super(EmptyComponent, self).__init__(
        spec=EmptyComponentSpec(), instance_name=name)


class CompilerUtilsTest(tf.test.TestCase):

  def testSetRuntimeParameterPb(self):
    pb = pipeline_pb2.RuntimeParameter()
    compiler_utils.set_runtime_parameter_pb(pb, "test_name", str,
                                            "test_default_value")
    print("testSetRuntimeParameterPb", pb)
    expected_pb = pipeline_pb2.RuntimeParameter(
        name="test_name",
        type=pipeline_pb2.RuntimeParameter.Type.STRING,
        default_value=metadata_store_pb2.Value(
            string_value="test_default_value"))
    self.assertEqual(expected_pb, pb)

  def testIsResolver(self):
    resolver = ResolverNode(
        instance_name="test_resolver_name",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver)
    self.assertTrue(compiler_utils.is_resolver(resolver))

    example_gen = CsvExampleGen(input=external_input("data_path"))
    self.assertFalse(compiler_utils.is_resolver(example_gen))

  def testIsComponent(self):
    resolver = ResolverNode(
        instance_name="test_resolver_name",
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver)
    self.assertFalse(compiler_utils.is_component(resolver))

    example_gen = CsvExampleGen(input=external_input("data_path"))
    self.assertTrue(compiler_utils.is_component(example_gen))

  def testEnsureTopologicalOrder(self):
    a = EmptyComponent(name="a")
    b = EmptyComponent(name="b")
    c = EmptyComponent(name="c")
    a.add_downstream_node(b)
    a.add_downstream_node(c)
    valid_orders = {"abc", "acb"}
    for order in itertools.permutations([a, b, c]):
      if "".join([c._instance_name for c in order]) in valid_orders:
        self.assertTrue(compiler_utils.ensure_topological_order(order))
      else:
        self.assertFalse(compiler_utils.ensure_topological_order(order))


if __name__ == "__main__":
  tf.test.main()
