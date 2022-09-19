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
"""Test pipeline for tfx.dsl.compiler.compiler."""
import os

from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.orchestration import pipeline


def create_producer_pipeline():
  example_gen = CsvExampleGen(input_base=os.path.join('tfx_root', 'data_path'))

  return pipeline.Pipeline(
      pipeline_name='producer-pipeline',
      components=[example_gen],
      outputs={'examples': example_gen.outputs['examples']})


def create_test_pipeline():
  """Builds a consumer pipeline that requires artifacts from producer pipelines."""
  producer_pipeline = create_producer_pipeline()

  pipeline_inputs = pipeline.PipelineInputs(
      {'examples': producer_pipeline.outputs['examples']})

  example_gen_resolver = resolver.Resolver(
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
      examples=pipeline_inputs['examples']).with_id(
          'Resolver.example_gen_resolver')

  statistics_gen = StatisticsGen(
      examples=example_gen_resolver.outputs['examples'])

  return pipeline.Pipeline(
      inputs=pipeline_inputs,
      pipeline_name='consumer-pipeline',
      components=[example_gen_resolver, statistics_gen])
