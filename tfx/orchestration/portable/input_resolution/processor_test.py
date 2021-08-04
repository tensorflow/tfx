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
"""Tests for tfx.orchestration.portable.input_resolution.processor."""
import importlib
import sys
from unittest import mock

import tensorflow as tf
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import processor
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from google.protobuf import text_format


class RepeatStrategy(resolver.ResolverStrategy):

  def __init__(self, num: int):
    self.num = num
    self.call_history = []
    RepeatStrategy.last_created = self

  def resolve_artifacts(self, store, input_dict):
    self.call_history.append((store, input_dict))
    return {key: value * self.num for key, value in input_dict.items()}


class NoneStrategy(resolver.ResolverStrategy):

  def resolve_artifacts(self, store, input_dict):
    return None


class RepeatOp(resolver_op.ResolverOp):
  num = resolver_op.ResolverOpProperty(type=int)

  def __init__(self):
    RepeatOp.last_created = self

  def apply(self, input_dict):
    return {key: value * self.num for key, value in input_dict.items()}


_original_import_module = importlib.import_module


def _import_module(module_path: str):
  if module_path == (
      'tfx.orchestration.portable.input_resolution.processor_test'):
    return sys.modules[__name__]
  else:
    return _original_import_module(module_path)


class ProcessorTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._input_dict = {
        'examples': [standard_artifacts.Examples()],
        'model': [standard_artifacts.Model()],
    }
    self._store = mock.Mock()
    self.enter_context(mock.patch('importlib.import_module', _import_module))

  def testRunResolverSteps_ResolverStrategy(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    strategy = RepeatStrategy.last_created
    self.assertIs(strategy.call_history[0][0], self._store)
    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testRunResolverSteps_ChainedResolverStrategies(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)

  def testRunResolverSteps_ResolverStrategy_HandleInputKeys(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatStrategy"
      config_json: "{\"num\": 2}"
      input_keys: ["examples"]
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 1)

  def testRunResolverSteps_NoneRaisesSignal(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse("""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.NoneStrategy"
    }
    """, config)
    with self.assertRaises(exceptions.InputResolutionError):
      processor.run_resolver_steps(
          self._input_dict,
          resolver_steps=config.resolver_steps,
          store=self._store)

  def testRunResolverSteps_ResolverOp(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    op = RepeatOp.last_created
    self.assertIs(op.context.store, self._store)
    self.assertEqual(op.num, 2)
    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testRunResolverSteps_ChainedResolverOps(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)

  def testRunResolverSteps_ResolverOp_IgnoresInputKeys(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatOp"
      config_json: "{\"num\": 2}"
      input_keys: ["examples"]
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    self.assertLen(result['examples'], 2)
    self.assertLen(result['model'], 2)

  def testRunResolverSteps_MixedResolverOpAndStrategy(self):
    config = pipeline_pb2.ResolverConfig()
    text_format.Parse(r"""
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatStrategy"
      config_json: "{\"num\": 2}"
    }
    resolver_steps {
      class_path: "tfx.orchestration.portable.input_resolution.processor_test.RepeatOp"
      config_json: "{\"num\": 2}"
    }
    """, config)
    result = processor.run_resolver_steps(
        self._input_dict,
        resolver_steps=config.resolver_steps,
        store=self._store)

    self.assertLen(result['examples'], 4)
    self.assertLen(result['model'], 4)


if __name__ == '__main__':
  tf.test.main()
