# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.input_resolution.resolver_config_resolver."""
from unittest import mock

import tensorflow as tf
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import resolver_config_resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format


@ops.testonly_register
class RepeatStrategy(resolver.ResolverStrategy):

  def __init__(self, num: int):
    self.num = num
    self.call_history = []
    RepeatStrategy.last_created = self

  def resolve_artifacts(self, store, input_dict):
    self.call_history.append((store, input_dict))
    return {key: value * self.num for key, value in input_dict.items()}


@ops.testonly_register
class NoneStrategy(resolver.ResolverStrategy):

  def resolve_artifacts(self, store, input_dict):
    return None


class UnregisteredStrategy(resolver.ResolverStrategy):

  def resolve_artifacts(self, store, input_dict):
    return input_dict


@ops.testonly_register
class RepeatOp(resolver_op.ResolverOp):
  num = resolver_op.Property(type=int)

  def __init__(self):
    RepeatOp.last_created = self

  def apply(self, input_dict):
    return {key: value * self.num for key, value in input_dict.items()}


class ResolverConfigResolverTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._input_dict = {
        'examples': [standard_artifacts.Examples()],
        'model': [standard_artifacts.Model()],
    }
    self._store = mock.Mock()

  def testResolve_ResolverStrategy(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    strategy = RepeatStrategy.last_created
    self.assertIs(strategy.call_history[0][0], self._store)
    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testResolve_ChainedResolverStrategies(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "__main__.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)

  def testResolve_ResolverStrategy_HandleInputKeys(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatStrategy"
      config_json: "{\"num\": 2}"
      input_keys: ["examples"]
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 1)

  def testResolve_NoneRaisesInputResolutionError(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse("""
    resolver_steps {
      class_path: "__main__.NoneStrategy"
    }
    """, config)
    with self.assertRaises(exceptions.InputResolutionError):
      resolver_config_resolver.resolve(
          self._store, self._input_dict, config)

  def testResolve_ResolverOp(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    op = RepeatOp.last_created
    self.assertIs(op.context.store, self._store)
    self.assertEqual(op.num, 2)
    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testResolve_ChainedResolverOps(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "__main__.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)

  def testResolve_ResolverOp_IgnoresInputKeys(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatOp"
      config_json: "{\"num\": 2}"
      input_keys: ["examples"]
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testResolve_MixedResolverOpAndStrategy(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "__main__.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)

  def testResolve_UnregisteredResolverStrategy(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "__main__.UnregisteredStrategy"
    }
    """, config)
    result = resolver_config_resolver.resolve(
        self._store, self._input_dict, config)

    self.assertEqual(result, self._input_dict)


if __name__ == '__main__':
  tf.test.main()
