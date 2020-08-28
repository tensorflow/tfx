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
"""Beam Component Fusion optimization test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf
from tfx.components import CsvExampleGen
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.experimental.optimizations.beam_fusion import BeamFusionOptimizer
from tfx.orchestration.experimental.optimizations.fused_component.component import FusedComponent
from tfx.orchestration.experimental.optimizations.fused_component.executor import Executor


class TestBeamFusionOptimization(tf.test.TestCase):

  def setUp(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), '../../components/testdata')
    self.csv_path = os.path.join(source_data_dir, 'external/csv')
    self.pipeline_root = tempfile.mkdtemp()
    super(TestBeamFusionOptimization, self).setUp()

  def testPartialPipelineOptimization(self):
    example_gen = CsvExampleGen(input_base=self.csv_path)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    pipeline_name = 'my_pipeline'
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self.pipeline_root,
        components=[example_gen, statistics_gen, schema_gen],
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            os.path.join(self.pipeline_root, 'metadata.db')))

    p.pipeline_info = data_types.PipelineInfo(
        pipeline_name=pipeline_name,
        pipeline_root=self.pipeline_root,
        run_id='my_run_id')
    optimizer = BeamFusionOptimizer(p)
    optimizer.optimize_pipeline()

    BeamDagRunner().run(p)

  def testFusedComponentExecutor(self):
    example_gen = CsvExampleGen(input_base=self.csv_path)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    subgraph = [example_gen, statistics_gen]

    fused_component = FusedComponent(
        subgraph=subgraph,
        beam_pipeline_args=[],
        pipeline_root=self.pipeline_root)
    executor = Executor()
    executor.Do(
        input_dict=fused_component.inputs,
        output_dict=fused_component.outputs,
        exec_properties=fused_component.exec_properties)


if __name__ == '__main__':
  tf.test.main()
