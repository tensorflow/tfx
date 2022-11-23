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

from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils
from tfx.utils import typing_utils

import ml_metadata as mlmd


class LatestPipelineRunOutputs(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPipelineRunOutputs',
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """LatestPipelineRunOutputs operator.

    This operator returns artifacts from the latest COMPLETED pipeline run.
  """

  pipeline_name = resolver_op.Property(type=str)
  output_keys = resolver_op.Property(type=Sequence[str], default=())

  def apply(self) -> typing_utils.ArtifactMultiMap:
    """Returns artifacts from the latest pipeline run.

    Returns:
      A dictionary, each value in the dict is a list of artifacts from the
      latest pipeline run.
    """
    if not self.pipeline_name:
      raise ValueError(
          'pipeline_name for LatestPipelineRunOutputs can not be empty.')

    # Gets the pipeline end node context.
    pipeline_end_node_name = compiler_utils.node_context_name(
        self.pipeline_name,
        compiler_utils.pipeline_end_node_id_from_pipeline_id(
            self.pipeline_name))
    pipeline_end_node_ctx = self.context.store.get_context_by_type_and_name(
        type_name=constants.NODE_CONTEXT_TYPE_NAME,
        context_name=pipeline_end_node_name)
    if not pipeline_end_node_ctx:
      raise exceptions.SkipSignal(
          f'Pipeline {self.pipeline_name} does not have a PipelineEnd node, '
          'possibly due to not defining the pipeline outputs.')

    # Gets the COMPLETE executions of the pipeline end node, and then find the
    # latest one.
    pipeline_end_node_executions = self.context.store.get_executions_by_context(
        context_id=pipeline_end_node_ctx.id,
        list_options=mlmd.ListOptions(
            filter_query='last_known_state = COMPLETE'))
    if not pipeline_end_node_executions:
      raise exceptions.SkipSignal(
          f'Pipeline {self.pipeline_name} does not have a successful execution.'
      )
    latest_execution = max(
        pipeline_end_node_executions,
        key=lambda e: (e.create_time_since_epoch, e.id))

    # From the latest execution, find out the latest artifacts.
    end_node_output_events = [
        e for e in self.context.store.get_events_by_execution_ids(
            execution_ids=[latest_execution.id])
        if event_lib.is_valid_output_event(e)
    ]
    if not end_node_output_events:
      raise exceptions.SkipSignal(
          f'Pipeline {self.pipeline_name} does not have any output artifacts '
          'from PipelineEnd node.')
    artifacts = self.context.store.get_artifacts_by_id(
        [e.artifact_id for e in end_node_output_events])
    artifact_types = self.context.store.get_artifact_types_by_id(
        list({a.type_id for a in artifacts}))
    artifact_types_by_id = {t.id: t for t in artifact_types}
    tfx_artifacts = [
        artifact_utils.deserialize_artifact(
            artifact_types_by_id[a.type_id], a) for a in artifacts]
    result = event_lib.reconstruct_artifact_multimap(
        tfx_artifacts, end_node_output_events)
    if self.output_keys:
      for key in list(result):
        if key not in self.output_keys:
          del result[key]
    return result
