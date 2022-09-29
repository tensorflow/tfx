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

from typing import Optional, Sequence

from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRun(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPipelineRun',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """LatestPipelineRun operator."""

  def _get_pipeline_run_context(
      self, pipeline_run_context_type_id: int,
      artifact: types.Artifact) -> Optional[metadata_store_pb2.Context]:
    # Gets the output event of the artifact.
    events = [
        e for e in self.context.store.get_events_by_artifact_ids([artifact.id])
        if e.type == metadata_store_pb2.Event.OUTPUT
    ]
    if not events:
      return None
    if len(events) > 1:
      raise ValueError('An artifact can not have more than one OUTPUT event.')

    # Gets the pipeline run context of the artifact.
    pipeline_run_contexts = [
        c for c in self.context.store.get_contexts_by_execution(
            events[0].execution_id) if c.type_id == pipeline_run_context_type_id
    ]
    if not pipeline_run_contexts:
      return None
    if len(pipeline_run_contexts) > 1:
      raise ValueError(
          'An execution can not have more than one pipeline run context.')

    return pipeline_run_contexts[0]

  def _select_latest_pipeline_run_artifact(
      self, input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    if not input_list:
      return []

    pipeline_run_context_type = self.context.store.get_context_type(
        type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)

    # Gets pipeline run contexts of the input artifacts.
    pipeline_run_contexts = []
    for a in input_list:
      pipeline_run_context = self._get_pipeline_run_context(
          pipeline_run_context_type.id, a)
      if pipeline_run_context:
        pipeline_run_contexts.append(pipeline_run_context)

    # Finds out the artifacts which belongs to the latest pipeline run context.
    pipeline_run_contexts.sort(  # pytype: disable=attribute-error
        key=lambda c: (c.create_time_since_epoch, c.id),
        reverse=True)
    for pipeline_run_context in pipeline_run_contexts:
      latest_pipeline_run_artifact_ids = [
          a.id for a in self.context.store.get_artifacts_by_context(
              pipeline_run_context.id)
      ]
      latest_pipeline_run_artifact = [
          a for a in input_list if a.id in latest_pipeline_run_artifact_ids
      ]
      if latest_pipeline_run_artifact:
        return latest_pipeline_run_artifact

    return []

  def apply(
      self, input_dict: typing_utils.ArtifactMultiMap
  ) -> typing_utils.ArtifactMultiMap:
    """Returns the artifacts from the latest pipeline run.

    For example, if the input_dict is {'key': [artifact_1, artifact_2]}.
    artifact_1 is from an earlier pipeline run, and artifact_2 is from a later
    pipeline run. The return value of this function is {'key': [artifact_2]}.

    Args:
      input_dict: A dictionary, each value in the dict is a list of artifacts.

    Returns:
      A dictionary, each value in the dict is a list of artifacts from the
      latest pipeline run.
    """
    if not input_dict:
      return {}

    return {
        key: self._select_latest_pipeline_run_artifact(value)
        for key, value in input_dict.items()
    }
