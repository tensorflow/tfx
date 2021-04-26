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
"""E2E Tests for tfx.examples.flowers_GAN.flowers_pipeline_local."""

import os
from typing import List, Text
import unittest

from absl import logging
from absl.testing import parameterized

import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.dsl.io import fileio
from tfx.examples.flowers_GAN import flowers_pipeline_local
from tfx.orchestration import metadata
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2
from tfx.proto import range_config_pb2
from tfx.utils import io_utils

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class FlowersPipelineLocalEndToEndTest(tf.test.TestCase,
                                       parameterized.TestCase):

    def setUp(self):
        super(FlowersPipelineLocalEndToEndTest, self).setUp()

        self._test_dir = os.path.join(
            os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
            self._testMethodName)

        self._pipeline_name = 'flowers_test'
        self._data_root = os.path.join(os.path.dirname(__file__), 'data')

        self._data_root = os.path.join(os.path.dirname(__file__), 'data')

        self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
        self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                           self._pipeline_name)
        self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                           self._pipeline_name, 'metadata.db')

    def _module_file_name(self, model_framework: str) -> str:
        return os.path.join(
            os.path.dirname(__file__), f'flowers_utils_{model_framework}.py')

    def _assertExecutedOnce(self, component: Text) -> None:
        """Check the component is executed exactly once."""
        component_path = os.path.join(self._pipeline_root, component)
        self.assertTrue(fileio.exists(component_path))
        execution_path = os.path.join(
            component_path, '.system', 'executor_execution')
        execution = fileio.listdir(execution_path)
        self.assertLen(execution, 1)

    def _assertPipelineExecution(self) -> None:
        self._assertExecutedOnce('ImportExampleGen')
        self._assertExecutedOnce('ExampleValidator')
        self._assertExecutedOnce('Pusher')
        self._assertExecutedOnce('SchemaGen')
        self._assertExecutedOnce('StatisticsGen')
        self._assertExecutedOnce('Trainer')
        self._assertExecutedOnce('Transform')

    @parameterized.parameters(
        ('keras',))
    def testFlowersPipelineLocal(self, model_framework):
        module_file = self._module_file_name(model_framework)
        pipeline = flowers_pipeline_local._create_pipeline(
            pipeline_name=self._pipeline_name,
            pipeline_root=self._pipeline_root,
            data_root=self._data_root,
            module_file=module_file,
            serving_model_dir=self._serving_model_dir,
            metadata_path=self._metadata_path,
            train_steps=5,
            eval_steps=5,
            examplegen_input_config=None,
            examplegen_range_config=None,
            beam_pipeline_args=[])

        logging.info('Starting the first pipeline run.')
        LocalDagRunner().run(pipeline)

        self.assertTrue(fileio.exists(self._serving_model_dir))
        self.assertTrue(fileio.exists(self._metadata_path))
        expected_execution_count = 7  # 7 components
        metadata_config = metadata.sqlite_metadata_connection_config(
            self._metadata_path)
        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(expected_execution_count, execution_count)

        self._assertPipelineExecution()

        logging.info('Starting the second pipeline run. '
                     'All components will use cached results.')
        LocalDagRunner().run(pipeline)

        # Asserts cache execution.
        with metadata.Metadata(metadata_config) as m:
            # Artifact count is unchanged.
            self.assertLen(m.store.get_artifacts(), artifact_count)
            self.assertLen(m.store.get_executions(), expected_execution_count * 2)

    def _get_input_examples_artifacts(
            self, store: mlmd.MetadataStore,
            execution_type: Text) -> List[metadata_store_pb2.Artifact]:
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


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
