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
"""Sample pipeline with ForEach context."""

from tfx.components import CsvExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.dsl.components.common import resolver
from tfx.dsl.control_flow import for_each
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2


def create_test_pipeline():
  """Creates a sample pipeline with ForEach context."""

  example_gen = CsvExampleGen(input_base='/data/mydummy_dataset')

  with for_each.ForEach(example_gen.outputs['examples']) as each_example:
    statistics_gen = StatisticsGen(examples=each_example)

  latest_stats_resolver = resolver.Resolver(
      statistics=statistics_gen.outputs['statistics'],
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
  ).with_id('latest_stats_resolver')

  schema_gen = SchemaGen(statistics=latest_stats_resolver.outputs['statistics'])

  with for_each.ForEach(example_gen.outputs['examples']) as each_example:
    trainer = Trainer(
        module_file='/src/train.py',
        examples=each_example,
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=2000),
    )

  with for_each.ForEach(trainer.outputs['model']) as each_model:
    pusher = Pusher(
        model=each_model,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='/models')),
    )

  return pipeline.Pipeline(
      pipeline_name='foreach',
      pipeline_root='/tfx/pipelines/foreach',
      components=[
          example_gen,
          statistics_gen,
          latest_stats_resolver,
          schema_gen,
          trainer,
          pusher,
      ],
      enable_cache=True,
      execution_mode=pipeline.ExecutionMode.SYNC,
  )
