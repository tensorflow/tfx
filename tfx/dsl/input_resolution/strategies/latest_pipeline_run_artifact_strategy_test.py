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
"""Test for LatestPipelineRunStrategy."""

import tensorflow as tf
from tfx.dsl.input_resolution.strategies import latest_pipeline_run_artifact_strategy as strategy
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRunStrategyTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._metadata = self.enter_context(
        metadata.Metadata(connection_config=self._connection_config))

  def testStrategy(self):
    store = self._metadata.store

    # Prepare artifacts, executions, and events.
    input_artifact_1 = standard_artifacts.Examples()
    input_artifact_1.uri = 'example'
    input_artifact_1.type_id = common_utils.register_type_if_not_exist(
        self._metadata, input_artifact_1.artifact_type).id
    input_artifact_2 = standard_artifacts.Examples()
    input_artifact_2.uri = 'example'
    input_artifact_2.type_id = common_utils.register_type_if_not_exist(
        self._metadata, input_artifact_2.artifact_type).id
    [input_artifact_1.id, input_artifact_2.id] = store.put_artifacts(
        [input_artifact_1.mlmd_artifact, input_artifact_2.mlmd_artifact])

    execution_type = metadata_store_pb2.ExecutionType(name='Example')
    execution_type_id = store.put_execution_type(execution_type)
    execution_1 = metadata_store_pb2.Execution(type_id=execution_type_id)
    execution_2 = metadata_store_pb2.Execution(type_id=execution_type_id)
    [execution_1_id,
     execution_2_id] = store.put_executions([execution_1, execution_2])

    output_event_1 = metadata_store_pb2.Event(
        artifact_id=input_artifact_1.id,
        execution_id=execution_1_id,
        type=metadata_store_pb2.Event.OUTPUT)
    output_event_2 = metadata_store_pb2.Event(
        artifact_id=input_artifact_2.id,
        execution_id=execution_2_id,
        type=metadata_store_pb2.Event.OUTPUT)
    store.put_events([output_event_1, output_event_2])

    # Prepare pipeline contexts.
    pipeline_ctx_type = metadata_store_pb2.ContextType(name='pipeline')
    ctx_type_id = store.put_context_type(pipeline_ctx_type)
    pipeline_ctx = metadata_store_pb2.Context(
        type_id=ctx_type_id, name='pipeline')
    [pipeline_ctx.id] = store.put_contexts([pipeline_ctx])

    # Prepare contexts, and link them to the artifacts and executions.
    run_ctx_type = metadata_store_pb2.ContextType(name='pipeline_run')
    run_ctx_type_id = store.put_context_type(run_ctx_type)
    pipeline_run_ctx_1 = metadata_store_pb2.Context(
        type_id=run_ctx_type_id, name='run-20220825-175117-371960')
    [pipeline_run_ctx_1.id] = store.put_contexts([pipeline_run_ctx_1])
    pipeline_run_ctx_2 = metadata_store_pb2.Context(
        type_id=run_ctx_type_id, name='run-20220825-175117-371961')
    [pipeline_run_ctx_2.id] = store.put_contexts([pipeline_run_ctx_2])

    attribution_1 = metadata_store_pb2.Attribution(
        artifact_id=input_artifact_1.id, context_id=pipeline_run_ctx_1.id)
    attribution_2 = metadata_store_pb2.Attribution(
        artifact_id=input_artifact_2.id, context_id=pipeline_run_ctx_2.id)
    attribution_3 = metadata_store_pb2.Attribution(
        artifact_id=input_artifact_1.id, context_id=pipeline_ctx.id)
    attribution_4 = metadata_store_pb2.Attribution(
        artifact_id=input_artifact_2.id, context_id=pipeline_ctx.id)
    association_1 = metadata_store_pb2.Association(
        execution_id=execution_1_id, context_id=pipeline_ctx.id)
    association_2 = metadata_store_pb2.Association(
        execution_id=execution_2_id, context_id=pipeline_ctx.id)
    store.put_attributions_and_associations(
        [attribution_1, attribution_2, attribution_3, attribution_4],
        [association_1, association_2])
    store.put_parent_contexts([
        metadata_store_pb2.ParentContext(
            child_id=pipeline_run_ctx_1.id, parent_id=pipeline_ctx.id),
        metadata_store_pb2.ParentContext(
            child_id=pipeline_run_ctx_2.id, parent_id=pipeline_ctx.id)
    ])

    # Run the function for test.
    latest_pipeline_run_strategy = strategy.LatestPipelineRunArtifactStrategy()
    result = latest_pipeline_run_strategy.resolve_artifacts(
        store, {'input': [input_artifact_1, input_artifact_2]})

    # Test the results.
    expected_artifact = [input_artifact_2]
    self.assertIsNotNone(result)
    self.assertEqual(expected_artifact, result['input'])


if __name__ == '__main__':
  tf.test.main()
