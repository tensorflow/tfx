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
"""TFX CsvExampleGen component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from tfx.components.example_gen import component
from tfx.components.example_gen.csv_example_gen import executor
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2


class CsvExampleGen(component.FileBasedExampleGen):  # pylint: disable=protected-access
  """Official TFX CsvExampleGen component.

  The csv examplegen component takes csv data, and generates train
  and eval examples for downstream components.

  The csv examplegen encodes column values to tf.Example int/float/byte feature.
  For the case when there's missing cells, the csv examplegen uses:
  -- tf.train.Feature(`type`_list=tf.train.`type`List(value=[])), when the
     `type` can be inferred.
  -- tf.train.Feature() when it cannot infer the `type` from the column.

  Note that the type inferring will be per input split. If input isn't a single
  split, users need to ensure the column types align in each pre-splits.

  For example, given the following csv rows of a split:

    header:A,B,C,D
    row1:  1,,x,0.1
    row2:  2,,y,0.2
    row3:  3,,,0.3
    row4:

  The output example will be
    example1: 1(int), empty feature(no type), x(string), 0.1(float)
    example2: 2(int), empty feature(no type), x(string), 0.2(float)
    example3: 3(int), empty feature(no type), empty list(string), 0.3(float)

    Note that the empty feature is `tf.train.Feature()` while empty list string
    feature is `tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))`.

  Component `outputs` contains:
   - `examples`: Channel of type `standard_artifacts.Examples` for output train
                 and eval examples.
  """

  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(self,
               input_base: Optional[Text] = None,
               input_config: Optional[Union[example_gen_pb2.Input,
                                            Dict[Text, Any]]] = None,
               output_config: Optional[Union[example_gen_pb2.Output,
                                             Dict[Text, Any]]] = None,
               range_config: Optional[Union[range_config_pb2.RangeConfig,
                                            Dict[Text, Any]]] = None):
    """Construct a CsvExampleGen component.

    Args:
      input_base: an external directory containing the CSV files.
      input_config: An example_gen_pb2.Input instance, providing input
        configuration. If unset, the files under input_base will be treated as a
        single split. If any field is provided as a RuntimeParameter,
        input_config should be constructed as a dict with the same field names
        as Input proto message.
      output_config: An example_gen_pb2.Output instance, providing output
        configuration. If unset, default splits will be 'train' and 'eval' with
        size 2:1. If any field is provided as a RuntimeParameter, output_config
        should be constructed as a dict with the same field names as Output
        proto message.
      range_config: An optional range_config_pb2.RangeConfig instance,
        specifying the range of span values to consider. If unset, driver will
        default to searching for latest span with no restrictions.
    """
    super(CsvExampleGen, self).__init__(
        input_base=input_base,
        input_config=input_config,
        output_config=output_config,
        range_config=range_config)
