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
"""E2E Tests for tfx.examples.penguin.penguin_pipeline_local."""

import os
from typing import List
import unittest

from absl import logging
from absl.testing import parameterized
import tensorflow as tf
from tfx.examples.penguin import penguin_pipeline_local
from tfx.v1 import proto
from tfx.v1.dsl.io import fileio
from tfx.v1.orchestration import LocalDagRunner
from tfx.v1.orchestration import metadata

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

_SPAN_PROPERTY_NAME = 'span'


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class PenguinPipelineLocalEndToEndTest(tf.test.TestCase,
                                       parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'penguin_test'
    self._schema_path = os.path.join(
        os.path.dirname(__file__), 'schema', 'user_provided', 'schema.pbtxt')
    self._data_root = os.path.join(os.path.dirname(__file__), 'data')

    # Create a data root for rolling window test
    # - data/labelled
    #   - day1
    #     - penguins_processed.csv
    #   - day2
    #     - penguins_processed.csv
    #   - day3
    #     - penguins_processed.csv
    self._data_root_span = os.path.join(self._test_dir, 'data')
    for day in ['day1', 'day2', 'day3']:
      dst_path = os.path.join(self._data_root_span, 'labelled', day)
      fileio.makedirs(dst_path)
      fileio.copy(
          os.path.join(self._data_root, 'labelled', 'penguins_processed.csv'),
          os.path.join(dst_path, 'penguins_processed.csv'))

    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def _module_file_name(self, model_framework: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), f'penguin_utils_{model_framework}.py')

  def _assertExecutedOnce(self, component: str) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    execution_path = os.path.join(
        component_path, '.system', 'executor_execution')
    execution = fileio.listdir(execution_path)
    self.assertLen(execution, 1)

  def _assertPipelineExecution(self,
                               has_tuner: bool = False,
                               has_bulk_inferrer: bool = False,
                               has_schema_gen: bool = True,
                               has_pusher: bool = True) -> None:
    self._assertExecutedOnce('CsvExampleGen')
    self._assertExecutedOnce('Evaluator')
    self._assertExecutedOnce('StatisticsGen')
    self._assertExecutedOnce('Trainer')
    self._assertExecutedOnce('Transform')
    if has_schema_gen:
      self._assertExecutedOnce('SchemaGen')
    else:
      self._assertExecutedOnce('ImportSchemaGen')
      self._assertExecutedOnce('ExampleValidator')
    if has_tuner:
      self._assertExecutedOnce('Tuner')
    if has_bulk_inferrer:
      self._assertExecutedOnce('CsvExampleGen_Unlabelled')
      self._assertExecutedOnce('BulkInferrer')
    if has_pusher:
      self._assertExecutedOnce('Pusher')

  def _make_beam_pipeline_args(self):
    return []

  @parameterized.parameters(
      ('keras',),
      ('flax_experimental',))
  def testPenguinPipelineLocal(self, model_framework):
    module_file = self._module_file_name(model_framework)
    pipeline = penguin_pipeline_local._create_pipeline(
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=module_file,
        accuracy_threshold=0.1,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        user_provided_schema_path=None,
        enable_tuning=False,
        enable_bulk_inferrer=False,
        examplegen_input_config=None,
        examplegen_range_config=None,
        resolver_range_config=None,
        beam_pipeline_args=self._make_beam_pipeline_args(),
        enable_transform_input_cache=False)

    logging.info('Starting the first pipeline run.')
    LocalDagRunner().run(pipeline)

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 8  # 7 components + 1 resolver
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)

    self._assertPipelineExecution()

    logging.info('Starting the second pipeline run. All components except '
                 'Evaluator and Pusher will use cached results.')
    LocalDagRunner().run(pipeline)

    # Artifact count is increased by 3 caused by Evaluator and Pusher.
    self.assertLen(store.get_artifacts(), artifact_count + 3)
    artifact_count = len(store.get_artifacts())
    self.assertLen(store.get_executions(), expected_execution_count * 2)

    logging.info('Starting the third pipeline run. '
                 'All components will use cached results.')
    LocalDagRunner().run(pipeline)

    # Asserts cache execution.
    # Artifact count is unchanged.
    self.assertLen(store.get_artifacts(), artifact_count)
    self.assertLen(store.get_executions(), expected_execution_count * 3)

  def testPenguinPipelineLocalWithTuner(self):
    # TODO(b/180723394): Parameterize this test when Flax supports tuning.
    module_file = self._module_file_name('keras')
    LocalDagRunner().run(
        penguin_pipeline_local._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=None,
            enable_tuning=True,
            enable_bulk_inferrer=False,
            examplegen_input_config=None,
            examplegen_range_config=None,
            resolver_range_config=None,
            beam_pipeline_args=self._make_beam_pipeline_args(),
            enable_transform_input_cache=False))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 9  # 8 components + 1 resolver
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)

    self._assertPipelineExecution(has_tuner=True)

  @parameterized.parameters(('keras',), ('flax_experimental',))
  def testPenguinPipelineLocalWithBulkInferrer(self, model_framework):
    module_file = self._module_file_name(model_framework)
    LocalDagRunner().run(
        penguin_pipeline_local._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=None,
            enable_tuning=False,
            enable_bulk_inferrer=True,
            examplegen_input_config=None,
            examplegen_range_config=None,
            resolver_range_config=None,
            beam_pipeline_args=[],
            enable_transform_input_cache=False))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 10  # 9 components + 1 resolver
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)

    self._assertPipelineExecution(has_bulk_inferrer=True)

  def testPenguinPipelineLocalWithImporter(self):
    module_file = self._module_file_name('keras')
    LocalDagRunner().run(
        penguin_pipeline_local._create_pipeline(
            pipeline_name=self._pipeline_name,
            data_root=self._data_root,
            module_file=module_file,
            accuracy_threshold=0.1,
            serving_model_dir=self._serving_model_dir,
            pipeline_root=self._pipeline_root,
            metadata_path=self._metadata_path,
            user_provided_schema_path=self._schema_path,
            enable_tuning=False,
            enable_bulk_inferrer=False,
            examplegen_input_config=None,
            examplegen_range_config=None,
            resolver_range_config=None,
            beam_pipeline_args=[],
            enable_transform_input_cache=False))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 9  # 7 components + 1 resolver + 1 importer
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)

    self._assertPipelineExecution(has_schema_gen=False)

  def _get_input_examples_artifacts(
      self, store: mlmd.MetadataStore,
      execution_type: str) -> List[metadata_store_pb2.Artifact]:
    executions = store.get_executions_by_type(execution_type)
    # Get latest execution.
    execution = max(executions, key=lambda a: a.id)
    events = store.get_events_by_execution_ids([execution.id])
    artifact_ids = []
    for event in events:
      for step in event.path.steps:
        if step.key == 'examples':
          artifact_ids.append(event.artifact_id)
          break
    return store.get_artifacts_by_id(artifact_ids)

  def testPenguinPipelineLocalWithRollingWindow(self):
    module_file = self._module_file_name('keras')
    examplegen_input_config = proto.Input(splits=[
        proto.Input.Split(name='test', pattern='day{SPAN}/*'),
    ])
    resolver_range_config = proto.RangeConfig(
        rolling_range=proto.RollingRange(num_spans=2))

    def run_pipeline(examplegen_range_config):
      LocalDagRunner().run(
          penguin_pipeline_local._create_pipeline(
              pipeline_name=self._pipeline_name,
              data_root=self._data_root_span,
              module_file=module_file,
              accuracy_threshold=0.1,
              serving_model_dir=self._serving_model_dir,
              pipeline_root=self._pipeline_root,
              metadata_path=self._metadata_path,
              user_provided_schema_path=None,
              enable_tuning=False,
              enable_bulk_inferrer=False,
              examplegen_input_config=examplegen_input_config,
              examplegen_range_config=examplegen_range_config,
              resolver_range_config=resolver_range_config,
              beam_pipeline_args=self._make_beam_pipeline_args(),
              enable_transform_input_cache=True))

    # Trigger the pipeline for the first span.
    examplegen_range_config = proto.RangeConfig(
        static_range=proto.StaticRange(
            start_span_number=1, end_span_number=1))
    run_pipeline(examplegen_range_config)

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    self._assertPipelineExecution()
    transform_execution_type = 'tfx.components.transform.component.Transform'
    trainer_execution_type = 'tfx.components.trainer.component.Trainer'
    expected_execution_count = 10  # 7 components + 3 resolvers
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)
    # Verify Transform's input examples artifacts.
    tft_input_examples_artifacts = self._get_input_examples_artifacts(
        store, transform_execution_type)
    self.assertLen(tft_input_examples_artifacts, 1)
    # SpansResolver (controlled by resolver_range_config) returns span 1.
    self.assertEqual(
        1, tft_input_examples_artifacts[0].custom_properties[
            _SPAN_PROPERTY_NAME].int_value)

    # Trigger the pipeline for the second span.
    examplegen_range_config = proto.RangeConfig(
        static_range=proto.StaticRange(
            start_span_number=2, end_span_number=2))
    run_pipeline(examplegen_range_config)

    execution_count = len(store.get_executions())
    # In the second run, evaluator may not bless the model. As a result,
    # conditional-guarded Pusher may not be executed.
    expected_minimum_execution_count = expected_execution_count * 2 - 1
    self.assertLessEqual(expected_minimum_execution_count, execution_count)
    # Verify Transform's input examples artifacts.
    tft_input_examples_artifacts = self._get_input_examples_artifacts(
        store, transform_execution_type)
    self.assertLen(tft_input_examples_artifacts, 2)
    spans = {
        tft_input_examples_artifacts[0].custom_properties[
            _SPAN_PROPERTY_NAME].int_value,
        tft_input_examples_artifacts[1].custom_properties[
            _SPAN_PROPERTY_NAME].int_value
    }
    # SpansResolver (controlled by resolver_range_config) returns span 1 & 2.
    self.assertSetEqual({1, 2}, spans)
    # Verify Trainer's input examples artifacts.
    self.assertLen(
        self._get_input_examples_artifacts(store, trainer_execution_type),
        2)

    # Trigger the pipeline for the thrid span.
    examplegen_range_config = proto.RangeConfig(
        static_range=proto.StaticRange(
            start_span_number=3, end_span_number=3))
    run_pipeline(examplegen_range_config)

    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    execution_count = len(store.get_executions())
    # Similar to the second run, conditional-guarded Pusher may not be executed.
    expected_minimum_execution_count = expected_execution_count * 3 - 2
    self.assertLessEqual(expected_minimum_execution_count, execution_count)
    # Verify Transform's input examples artifacts.
    tft_input_examples_artifacts = self._get_input_examples_artifacts(
        store, transform_execution_type)
    self.assertLen(tft_input_examples_artifacts, 2)
    spans = {
        tft_input_examples_artifacts[0].custom_properties[
            _SPAN_PROPERTY_NAME].int_value,
        tft_input_examples_artifacts[1].custom_properties[
            _SPAN_PROPERTY_NAME].int_value
    }
    # SpansResolver (controlled by resolver_range_config) returns span 2 & 3.
    self.assertSetEqual({2, 3}, spans)
    # Verify Trainer's input examples artifacts.
    self.assertLen(
        self._get_input_examples_artifacts(store, trainer_execution_type),
        2)

  def testPenguinPipelineLocalConditionalWithoutPusher(self):
    module_file = self._module_file_name('keras')
    pipeline = penguin_pipeline_local._create_pipeline(
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=module_file,
        accuracy_threshold=1.0,  # Model evaluation will fail with 1.0 threshold
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        user_provided_schema_path=None,
        enable_tuning=False,
        enable_bulk_inferrer=False,
        examplegen_input_config=None,
        examplegen_range_config=None,
        resolver_range_config=None,
        beam_pipeline_args=self._make_beam_pipeline_args(),
        enable_transform_input_cache=False)

    logging.info('Starting the first pipeline run.')
    LocalDagRunner().run(pipeline)

    self.assertTrue(fileio.exists(self._metadata_path))
    expected_execution_count = 7  # Without pusher because evaluation fails
    metadata_config = metadata.sqlite_metadata_connection_config(
        self._metadata_path)
    store = mlmd.MetadataStore(metadata_config)
    artifact_count = len(store.get_artifacts())
    execution_count = len(store.get_executions())
    self.assertGreaterEqual(artifact_count, execution_count)
    self.assertEqual(expected_execution_count, execution_count)

    self._assertPipelineExecution(has_pusher=False)

    logging.info('Starting the second pipeline run. All components except '
                 'Evaluator will use cached results. Pusher will not run.')
    LocalDagRunner().run(pipeline)

    # Artifact count stays the same, because no new blessed model,
    # hence no new evaluation and no new pushed model.
    self.assertLen(store.get_artifacts(), artifact_count)
    self.assertLen(store.get_executions(), expected_execution_count * 2)

    logging.info('Starting the third pipeline run. '
                 'All components will use cached results.')
    LocalDagRunner().run(pipeline)

    # Asserts cache execution.
    # Artifact count is unchanged.
    self.assertLen(store.get_artifacts(), artifact_count)
    self.assertLen(store.get_executions(), expected_execution_count * 3)


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
