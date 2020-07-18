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
"""Iris flowers example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2

from tfx.components import CsvExampleGen
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.types.standard_artifacts import Schema
from tfx.types.standard_artifacts import Examples
from tfx.types.standard_artifacts import ExampleStatistics
from tfx.types.channel import Channel
from tfx.proto import example_gen_pb2
from tfx.proto import trainer_pb2

from tfx.utils import path_utils
from tfx.utils import io_utils
from tfx.utils.dsl_utils import external_input


def _create_example_pipeline(pipeline_name: Text, pipeline_root: Text,
                             data_root: Text, metadata_path: Text,
                             beam_pipeline_args: List[Text]
                             ) -> pipeline.Pipeline:
  """Simple pipeline to ingest data into Examples artifacts."""
  components = []

  # Brings data into the pipeline or otherwise joins/converts training data.
  input_config = example_gen_pb2.Input(splits=[
      example_gen_pb2.Input.Split(name='single_split', 
                                  pattern='span{SPAN}/*')])
  example_gen = CsvExampleGen(input_base=data_root,
                              input_config=input_config)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(
      examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen
      ],
      enable_cache=False,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)

def _create_trainer_pipeline(pipeline_name: Text, pipeline_root: Text,
                             module_file: Text, metadata_path: Text,
                             window_size: int, beam_pipeline_args: List[Text],
                             ) -> pipeline.Pipeline:
  """Trainer pipeline to train based on resolver outputs"""
  # Get latest schema for training.
  schema_resolver = ResolverNode(
      instance_name='schema_resolver',
      resolver_class=latest_artifacts_resolver.LatestArtifactsResolver,
      schema=Channel(type=Schema))

  # Resolve latest two example artifacts into one channel for trainer.
  latest_examples_resolver = ResolverNode(
      instance_name='latest_examples_resolver',
      resolver_class=latest_artifacts_resolver.LatestArtifactsResolver,
      resolver_configs={'desired_num_of_artifacts': window_size},
      latest_n_examples=Channel(type=Examples))

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=latest_examples_resolver.outputs['latest_n_examples'],
      schema=schema_resolver.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=2000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          schema_resolver,
          latest_examples_resolver,
          trainer
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


class IrisResolverEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(IrisResolverEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    # self._test_dir = os.path.join(os.path.dirname(__file__), 'TEST')

    self._pipeline_name = 'resolver_test'
    self._init_data_root = os.path.join(os.path.dirname(__file__), 'data')
    self._data_root = os.path.join(self._test_dir, 'data')
    self._module_file = os.path.join(os.path.dirname(__file__), 'iris_utils.py')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._window_size = 2

  def testIrisPipelineResolver(self):
    example_gen_pipeline = _create_example_pipeline(
        pipeline_name=self._pipeline_name,
        pipeline_root=self._pipeline_root,
        data_root=self._data_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])
    
    trainer_pipeline = _create_trainer_pipeline(
        pipeline_name=self._pipeline_name,
        pipeline_root=self._pipeline_root,
        module_file=self._module_file,
        metadata_path=self._metadata_path,
        window_size=self._window_size,
        beam_pipeline_args=[])

    # Generate two example artifacts.
    for i in range(self._window_size):
        io_utils.copy_file(os.path.join(self._init_data_root, 'iris.csv'),
                           os.path.join(self._data_root, 'span' + str(i+1),
                           'iris.csv'))
        BeamDagRunner().run(example_gen_pipeline)

    # Train on example artifacts, which are pulled using ResolverNode.
    BeamDagRunner().run(trainer_pipeline)

    # Test Trainer output.
    self.assertTrue(tf.io.gfile.exists(self._metadata_path))
    trainer_dir = os.path.join(self._pipeline_root, 'Trainer', 'model')
    working_dir = io_utils.get_only_uri_in_dir(trainer_dir)
    self.assertTrue(
        tf.io.gfile.exists(path_utils.serving_model_path(working_dir)))
    
    # Query MLMD to see if trainer and resolver_node worked properly.
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.filename_uri = self._metadata_path
    connection_config.sqlite.connection_mode = metadata_store_pb2.SqliteMetadataSourceConfig.READWRITE_OPENCREATE
    store = metadata_store.MetadataStore(connection_config)

    # Get example artifact ids.
    example_ids = [e.id for e in store.get_artifacts_by_type('Examples')]

    # Get latest example resolver execution information.
    all_resolvers = store.get_executions_by_type(
        'tfx.components.common_nodes.resolver_node.ResolverNode')
    resolver_exec = [e for e in all_resolvers 
        if e.properties['component_id'] == metadata_store_pb2.Value(
            string_value='ResolverNode.latest_examples_resolver')][0]

    # Check if window size is exactly equal to number of examples
    # appearing in output events from example resolver.
    resolver_events = store.get_events_by_execution_ids([resolver_exec.id])
    self.assertEqual(self._window_size,
        len([e for e in resolver_events if e.artifact_id in example_ids and
                e.type == metadata_store_pb2.Event.Type.OUTPUT]))
    
    # Get trainer component execution information.
    trainer_exec = store.get_executions_by_type(
        'tfx.components.trainer.component.Trainer')[0]

    # Check if window size is exactly equal to number of examples
    # appearing in input events to Trainer.
    train_events = store.get_events_by_execution_ids([trainer_exec.id])
    self.assertEqual(self._window_size,
        len([e for e in train_events if e.artifact_id in example_ids and
                e.type == metadata_store_pb2.Event.Type.INPUT]))


if __name__ == '__main__':
  tf.test.main()
