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
"""Beam Component Fusion optimization test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from fused_component.component import FusedComponent
from fused_component.executor import Executor

_pipeline_name = 'chicago_taxi_beam'

# This example assumes that the taxi data is stored in ~/taxi/data and the
# taxi utility function is in ~/taxi.  Feel free to customize this as needed.
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data', 'simple')

# Directory and data locations.  This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)


def test_fused_component_executor():
  example_gen = CsvExampleGen(input_base=_data_root)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  subgraph = [example_gen, statistics_gen]

  direct_num_workers = 0
  beam_pipeline_args = ['--direct_num_workers=%d' % direct_num_workers]

  fused_component = FusedComponent(subgraph=subgraph,
                                   beam_pipeline_args=beam_pipeline_args,
                                   pipeline_root=_pipeline_root)

  executor = Executor()
  executor.Do(input_dict=fused_component.inputs,
              output_dict=fused_component.outputs,
              exec_properties=fused_component.exec_properties)

if __name__ == '__main__':
  test_fused_component_executor()
