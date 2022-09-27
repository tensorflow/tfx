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

from typing import Mapping, Sequence, Tuple

from tfx import types
from tfx.dsl.compiler import constants
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils

import ml_metadata as mlmd
from ml_metadata import errors
from ml_metadata.proto import metadata_store_pb2


class LatestPipelineRun(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPipelineRun',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """LatestPipelineRun operator.

    This operator returns artifacts from the latest pipeline run that produces
    all the required keys.

    Example 1:
      result = latest_pipeline_run_op.apply(
          {'key': [artifact_run1, artifact_run2]}
      )

      If artifact_run1 is from the first run, and artifact_run2 is from the
      second run.
      The result is {'key': [artifact_run2]}.

    Example 2:
      result = latest_pipeline_run_op.apply(
          {'key1': [key1_artifact_run1, key1_artifact_run2],
           'key2': [key2_artifact_run1]}
      )

      If key1_artifact_run1 and key2_artifact_run1 are from the first run.
      key1_artifact_run2 is from the second run.
      The result is from the first run {'key1': [key1_artifact_run1], key2:
      [key2_artifact_run1]}. The artifacts from the second run is not returned
      because the second run does not have artifacts for key2.
  """

  _pipeline_run_context_type_id = None
  _node_context_type_id = None

  def _get_run_context_and_node_context(
      self, artifact: types.Artifact
  ) -> Tuple[metadata_store_pb2.Context, metadata_store_pb2.Context]:
    """Returns the run context and node context of the artifact."""
    # Gets the output event of the artifact.
    events = [
        e for e in self.context.store.get_events_by_artifact_ids([artifact.id])
        if e.type == metadata_store_pb2.Event.OUTPUT
    ]
    if not events:
      return (None, None)
    events.sort(key=lambda e: e.milliseconds_since_epoch)

    # Gets the run context and node context of the artifact.
    # We get contexts via execution since an execution only associates with no
    # more than one run context and one node context.
    contexts = self.context.store.get_contexts_by_execution(
        events[0].execution_id)
    pipeline_run_context = None
    node_context = None
    for c in contexts:
      if c.type_id == self._pipeline_run_context_type_id:
        pipeline_run_context = c
      if c.type_id == self._node_context_type_id:
        node_context = c
    if not pipeline_run_context or not node_context:
      return (None, None)

    return (pipeline_run_context, node_context)

  def _get_num_completed_executions(self, run_context_id: int,
                                    node_context_id: int) -> int:
    """Returns the number of executions associated with the run and node context.
    """
    filter_query = f'contexts_1.id = {run_context_id} AND contexts_2.id = {node_context_id} AND last_known_state = COMPLETE'
    executions = self.context.store.get_executions(
        list_options=mlmd.ListOptions(filter_query=filter_query))
    return len(executions)

  def _get_run_ctx_create_time_to_artifacts_mapping(
      self, input_list: Sequence[types.Artifact]
  ) -> Mapping[Tuple[int, int], Sequence[types.Artifact]]:
    if not input_list:
      return {}

    # Gets pipeline run and node contexts of the input artifacts.
    run_node_ctx_to_artifact = {}
    for artifact in input_list:
      (run_ctx, node_ctx) = self._get_run_context_and_node_context(artifact)
      run_node_ctx_to_artifact.setdefault(
          (run_ctx.create_time_since_epoch, run_ctx.id, node_ctx.id),
          []).append(artifact)

    # If the node is partially finished, i.e. the number of executions of the
    # node is not the same as the number of output artifacts, we skip the node.
    for key in list(run_node_ctx_to_artifact):
      num_completed_executions = self._get_num_completed_executions(
          key[1], key[2])
      if num_completed_executions != len(run_node_ctx_to_artifact[key]):
        del run_node_ctx_to_artifact[key]

    run_ctx_to_artifact = {}
    for key, artifacts in run_node_ctx_to_artifact.items():
      run_ctx_to_artifact.update({(key[0], key[1]): artifacts})

    return run_ctx_to_artifact

  def apply(
      self, input_dict: typing_utils.ArtifactMultiMap
  ) -> typing_utils.ArtifactMultiMap:
    """Returns artifacts from the latest pipeline run that produces all keys.

    Args:
      input_dict: A dictionary, each value in the dict is a list of artifacts.

    Returns:
      A dictionary, each value in the dict is a list of artifacts from the
      latest pipeline run.
    """
    if not input_dict:
      return {}

    # If any of the input value is empty, returns empty dict.
    for value in input_dict.values():
      if not value:
        return {}

    try:
      run_ctx_type = self.context.store.get_context_type(
          type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)
      self._pipeline_run_context_type_id = run_ctx_type.id
    except errors.NotFoundError:
      return {}
    try:
      node_ctx_type = self.context.store.get_context_type(
          type_name=constants.NODE_CONTEXT_TYPE_NAME)
      self._node_context_type_id = node_ctx_type.id
    except errors.NotFoundError:
      return {}

    # Constructs a mapping: Map[context_run_create_time: Map[key: [artifact]]]
    run_ctx_create_time_to_artifacts = {}
    for key, artifacts in input_dict.items():
      run_ctx_to_artifacts = self._get_run_ctx_create_time_to_artifacts_mapping(
          artifacts)
      for run_ctx, artifacts in run_ctx_to_artifacts.items():
        run_ctx_create_time_to_artifacts.setdefault(run_ctx,
                                                    {}).update({key: artifacts})

    # Sorts the pipeline run context by time, and then find out the latest
    # pipeline run that produces all keys.
    sorted_run_ctxs = sorted(
        list(run_ctx_create_time_to_artifacts.keys()), reverse=True)
    for run_ctx in sorted_run_ctxs:
      output_dict = run_ctx_create_time_to_artifacts[run_ctx]
      if len(output_dict) == len(input_dict):
        return output_dict

    return {}
