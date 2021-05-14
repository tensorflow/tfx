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
"""Tests for tfx.orchestration.portable.resolver.factory."""
import importlib

from absl.testing import parameterized
import tensorflow as tf
from tfx.dsl.components.common import resolver
from tfx.orchestration.portable.resolver import factory
from tfx.proto import range_config_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils


class FactoryTest(tf.test.TestCase, parameterized.TestCase):

  def class_path_exists(self, class_path):
    module_name, unused_class_name = class_path.rsplit('.', maxsplit=1)
    try:
      importlib.import_module(module_name)
    except ImportError:
      return False
    else:
      return True

  @parameterized.parameters(
      ('tfx.dsl.resolvers.oldest_artifacts_resolver'
       '.OldestArtifactsResolver',
       '{}'),
      ('tfx.dsl.resolvers.unprocessed_artifacts_resolver'
       '.UnprocessedArtifactsResolver',
       '{"execution_type_name": "Foo"}'),
      ('tfx.dsl.input_resolution.strategies.latest_artifact_strategy'
       '.LatestArtifactStrategy',
       '{}'),
      ('tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy'
       '.LatestBlessedModelStrategy',
       '{}'),
      ('tfx.dsl.input_resolution.strategies.span_range_strategy'
       '.SpanRangeStrategy',
       json_utils.dumps({
           'range_config': range_config_pb2.StaticRange(
               start_span_number=1, end_span_number=10)
       })),
      )
  def test_make_resolver_strategy_instance(self, class_path, config_json):
    if not self.class_path_exists(class_path):
      self.skipTest(f"Class path {class_path} doesn't exist.")
    resolver_step = pipeline_pb2.ResolverConfig.ResolverStep(
        class_path=class_path,
        config_json=config_json)

    result = factory.make_resolver_strategy_instance(resolver_step)

    self.assertIsInstance(result, resolver.ResolverStrategy)
    self.assertEndsWith(class_path, result.__class__.__name__)


if __name__ == '__main__':
  tf.test.main()
