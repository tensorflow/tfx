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
"""Tests for tfx.components.base.base_executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Dict, List, Text, Union, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import DirectOptions
from apache_beam.options.pipeline_options import StandardOptions
import tensorflow as tf

from tfx import types
from tfx.components.base import base_executor


_EXAMPLES = 'examples'
_SHUFFLED_EXAMPLES = 'shuffled_examples'
_NUM_LINES = 1000


class _TestExecutor(base_executor.BaseExecutor):
  """Fake executor for testing purpose only."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Union[int, float, Text]]) -> None:
    pass


class _BeamTestExecutor(base_executor.FuseableBeamExecutor):
  """Fake FuseableBeamExecutor for testing purpose only."""

  def beam_io_signature(
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Tuple[Dict[Text, type], Dict[Text, type]]:  # pylint: disable=g-bare-generic
    input_signature = {_EXAMPLES: Text}
    output_signature = {_SHUFFLED_EXAMPLES: int}
    return input_signature, output_signature

  def read_inputs(
      self, pipeline: beam.Pipeline, input_dict: Dict[Text,
                                                      List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Dict[Text, beam.pvalue.PCollection]:
    input_dir = input_dict[_EXAMPLES].uri
    beam_inputs = ({
        _EXAMPLES:
            (pipeline
             | beam.io.textio.ReadFromText(input_dir).with_output_types(Text))
    })
    return beam_inputs

  def run_component(
      self, pipeline: beam.Pipeline, beam_inputs: Dict[Text,
                                                       beam.pvalue.PCollection],
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]
  ) -> Dict[Text, beam.pvalue.PCollection]:
    beam_outputs = {
        _SHUFFLED_EXAMPLES: (beam_inputs[_EXAMPLES]
                             | beam.Map(str.strip)
                             | beam.Map(int)
                             | beam.CombineGlobally(sum))
    }
    return beam_outputs

  def write_outputs(
      self, pipeline: beam.Pipeline,
      beam_outputs: Dict[Text, beam.pvalue.PCollection],
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Union[int, float, Text]]) -> None:
    output_dir = output_dict[_SHUFFLED_EXAMPLES].uri
    _ = (
        beam_outputs[_SHUFFLED_EXAMPLES]
        | beam.io.textio.WriteToText(output_dir))

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Union[int, float, Text]]) -> None:
    with self._make_beam_pipeline() as p:
      beam_inputs = self.read_inputs(p, input_dict, output_dict,
                                     exec_properties)
      beam_outputs = self.run_component(p, beam_inputs, input_dict, output_dict,
                                        exec_properties)
      self.write_outputs(p, beam_outputs, input_dict, output_dict,
                         exec_properties)


class BaseExecutorTest(tf.test.TestCase):

  def testBeamSettings(self):
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=['--runner=DirectRunner'])
    executor = _TestExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(StandardOptions)
    self.assertEqual('DirectRunner', options.runner)

    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=['--direct_num_workers=2'])
    executor = _TestExecutor(executor_context)
    options = executor._make_beam_pipeline().options.view_as(DirectOptions)
    self.assertEqual(2, options.direct_num_workers)


class BeamExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(BeamExecutorTest, self).setUp()
    self._input_data_path = (
        os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            'input.txt'))
    output_data_path = (
        os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            'output.txt'))
    self._full_output_data_path = output_data_path + '-00000-of-00001'

    self._generate_data(self._input_data_path)

    # Create input dict
    examples = types.standard_artifacts.Examples()
    examples.uri = self._input_data_path
    self._input_dict = {_EXAMPLES: examples}

    # Create output dict
    shuffled_examples = types.standard_artifacts.Examples()
    shuffled_examples.uri = output_data_path
    self._output_dict = {_SHUFFLED_EXAMPLES: shuffled_examples}

  def _generate_data(self, file_path):
    with open(file_path, 'w+') as f:
      for i in range(_NUM_LINES):
        line = str(i) + '\n'
        f.write(line)

  def testFuseableBeamExecutorBeamIoSignature(self):
    beam_test_executor = _BeamTestExecutor()
    input_signature, output_signature = (
        beam_test_executor.beam_io_signature(self._input_dict,
                                             self._output_dict, {}))

    self.assertEqual(input_signature[_EXAMPLES], Text)
    self.assertEqual(output_signature[_SHUFFLED_EXAMPLES], int)

  def testFuseableBeamExecutorDo(self):
    beam_test_executor = _BeamTestExecutor()
    beam_test_executor.Do(self._input_dict, self._output_dict, {})

    expected_sum = 0
    for i in range(_NUM_LINES):
      expected_sum += i

    with open(self._full_output_data_path) as f:
      actual_sum = int(f.read())

    self.assertEqual(expected_sum, actual_sum)


if __name__ == '__main__':
  tf.test.main()
