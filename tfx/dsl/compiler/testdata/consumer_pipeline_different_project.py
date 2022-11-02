# Copyright 2022 Google LLC. All Rights Reserved.
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

from tfx.components import StatisticsGen
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.orchestration import pipeline
from tfx.types import channel_utils
from tfx.types import standard_artifacts


def create_test_pipeline():
  """Builds a consumer pipeline that gets artifacts from another project."""
  external_examples = channel_utils.external_project_artifact_query(
      artifact_type=standard_artifacts.Examples,
      project_owner='owner',
      project_name='producer-project',
      pipeline_name='producer-pipeline',
      producer_component_id='producer-component-id',
      output_key='output-key',
      pipeline_run_id='pipeline-run-id',
      mlmd_service_target='mlmd-service-target')

  example_gen_resolver = resolver.Resolver(
      strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
      examples=external_examples).with_id('Resolver.example_gen_resolver')

  statistics_gen = StatisticsGen(
      examples=example_gen_resolver.outputs['examples'])

  return pipeline.Pipeline(
      pipeline_name='consumer-pipeline',
      components=[example_gen_resolver, statistics_gen])
