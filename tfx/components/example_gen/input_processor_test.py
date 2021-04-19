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
"""Tests for tfx.components.example_gen.input_processor."""

from typing import Optional, Text
import tensorflow as tf
from tfx.components.example_gen import input_processor
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


class TestInputProcessor(input_processor.InputProcessor):

  def get_latest_span(self) -> int:
    return 0

  def get_pattern_for_span_version(self, pattern: Text, span: int,
                                   version: Optional[int]) -> Text:
    return ''


class InputProcessorTest(tf.test.TestCase):

  def testInputProcessor(self):
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='s', pattern='path/{SPAN}'),
    ])
    static_range_config = range_config_pb2.RangeConfig(
        static_range=range_config_pb2.StaticRange(
            start_span_number=1, end_span_number=2))
    rolling_range_config = range_config_pb2.RangeConfig(
        rolling_range=range_config_pb2.RollingRange(num_spans=2))
    rolling_range_config2 = range_config_pb2.RangeConfig(
        rolling_range=range_config_pb2.RollingRange(
            num_spans=1, start_span_number=1))

    with self.assertRaisesRegexp(
        ValueError,
        'For ExampleGen, start and end span numbers for RangeConfig.StaticRange must be equal'
    ):
      TestInputProcessor(input_config.splits, static_range_config)

    with self.assertRaisesRegexp(
        ValueError,
        'ExampleGen only support single span for RangeConfig.RollingRange'):
      TestInputProcessor(input_config.splits, rolling_range_config)

    with self.assertRaisesRegexp(
        ValueError,
        'RangeConfig.rolling_range.start_span_number is not supported'):
      TestInputProcessor(input_config.splits, rolling_range_config2)

  def testFileBasedInputProcessor(self):
    # TODO(b/181275944): migrate test after refactoring FileBasedInputProcessor.

    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='s1', pattern='path/{SPAN}'),
        example_gen_pb2.Input.Split(name='s2', pattern='path2')
    ])
    input_config2 = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='s', pattern='path'),
    ])

    static_range_config = range_config_pb2.RangeConfig(
        static_range=range_config_pb2.StaticRange(
            start_span_number=2, end_span_number=2))

    with self.assertRaisesRegexp(ValueError,
                                 'Spec setup should the same for all splits'):
      input_processor.FileBasedInputProcessor('input_base_uri',
                                              input_config.splits, None)

    with self.assertRaisesRegexp(ValueError,
                                 'Span or Date spec should be specified'):
      input_processor.FileBasedInputProcessor('input_base_uri',
                                              input_config2.splits,
                                              static_range_config)

  def testQueryBasedInputProcessor(self):
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='s', pattern='select * from table'),
    ])
    input_config_span = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(
            name='s1',
            pattern='select * from table where date=@span_yyyymmdd_utc'),
        example_gen_pb2.Input.Split(
            name='s2',
            pattern='select * from table2 where date=@span_yyyymmdd_utc')
    ])

    static_range_config = range_config_pb2.RangeConfig(
        static_range=range_config_pb2.StaticRange(
            start_span_number=2, end_span_number=2))
    rolling_range_config = range_config_pb2.RangeConfig(
        rolling_range=range_config_pb2.RollingRange(num_spans=1))

    with self.assertRaisesRegexp(
        NotImplementedError,
        'For QueryBasedExampleGen, latest Span is not supported'):
      processor = input_processor.QueryBasedInputProcessor(
          input_config_span.splits, rolling_range_config)
      processor.resolve_span_and_version()

    processor = input_processor.QueryBasedInputProcessor(
        input_config.splits, None)
    span, version = processor.resolve_span_and_version()
    fp = processor.get_input_fingerprint(span, version)
    self.assertEqual(span, 0)
    self.assertIsNone(version)
    self.assertIsNone(fp)

    processor = input_processor.QueryBasedInputProcessor(
        input_config_span.splits, static_range_config)
    span, version = processor.resolve_span_and_version()
    fp = processor.get_input_fingerprint(span, version)
    self.assertEqual(span, 2)
    self.assertIsNone(version)
    self.assertIsNone(fp)
    pattern = processor.get_pattern_for_span_version(
        input_config_span.splits[0].pattern, span, version)
    self.assertEqual(pattern, "select * from table where date='19700103'")


if __name__ == '__main__':
  tf.test.main()
