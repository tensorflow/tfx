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
"""Module for LatestPipelineRun operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.input_resolution import resolver_op

from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRun(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPipelineRun',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """LatestPipelineRun operator."""

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns the artifacts created in the latest pipeline run."""
    store = self.context.store

    if not input_list:
      return []

    events = [
        e for e in store.get_events_by_artifact_ids([input_list[0].id])
        if e.type == metadata_store_pb2.Event.OUTPUT
    ]
    if not events:
      return []

    # Get the pipeline context of the artifacts.
    pipeline_context_type = store.get_context_type(
        type_name=constants.PIPELINE_CONTEXT_TYPE_NAME)
    pipeline_contexts = [
        c for c in store.get_contexts_by_execution(events[0].execution_id)
        if c.type_id == pipeline_context_type.id
    ]
    if not pipeline_contexts:
      return []

    # Get the pipeline run contexts.
    pipeline_run_context_type = store.get_context_type(
        type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)
    pipeline_run_contexts = [
        c
        for c in store.get_children_contexts_by_context(pipeline_contexts[0].id)
        if c.type_id == pipeline_run_context_type.id
    ]
    if not pipeline_run_contexts:
      return []

    # Find out the artifacts created in the latest pipeline run.
    pipeline_run_contexts.sort(  # pytype: disable=attribute-error
        key=lambda c: (c.create_time_since_epoch, c.id),
        reverse=True)
    for pipeline_run_context in pipeline_run_contexts:
      latest_pipeline_run_artifact_ids = [
          a.id for a in store.get_artifacts_by_context(pipeline_run_context.id)
      ]
      latest_pipeline_run_artifact = [
          a for a in input_list if a.id in latest_pipeline_run_artifact_ids
      ]
      if latest_pipeline_run_artifact:
        return latest_pipeline_run_artifact

    return []
