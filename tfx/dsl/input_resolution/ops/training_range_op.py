# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Module for Shuffle operator."""

from typing import Any, List, Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils


def _validate_input_list(
    input_list: Sequence[types.Artifact],
) -> types.Artifact:
  """Checks that input_list contains only a single Model, and returns it."""
  if (
      len(input_list) > 1
      or input_list[0].TYPE_NAME != ops_utils.MODEL_TYPE_NAME
  ):
    raise exceptions.InvalidArgument(
        'The input_list for TrainingRange expects only a single Model artifact.'
    )

  return input_list[0]


def training_range(store: Any, model: types.Artifact) -> List[types.Artifact]:
  """ContainsTrainingRange implementation, for shared use across ResolverOps.

  Returns the Examples artifact the Model was trained on.

  Note that only the standard TFleX Model and Examples artifacts are supported.

  Args:
   store: The MetadataStore.
   model: The Model artifact whose trained Examples to return.

  Returns:
    List of Examples artifacts if found, else empty list. We intentionally don't
    raise SkipSignal, such that the caller can decide to raise it or not.
  """
  # In MLMD, an Examples and Model are related by:
  #
  #          Event 1           Event 2
  # Examples ------> Execution ------> Model
  #
  # For a single Model, there may be many parent Examples it was trained on.

  # TODO(kshivvy): Support querying multiple Model ids at once, to reduce the
  # number of round trip MLMD queries. This will be useful for resolving inputs
  # of a span driven evaluator.

  # Get all Executions associated with creating the Model.
  execution_ids = set()
  for event in store.get_events_by_artifact_ids([model.id]):
    if event_lib.is_valid_output_event(event):
      execution_ids.add(event.execution_id)

  # Get all artifact ids associated with an INPUT Event in each Execution.
  # These ids correspond to parent artifacts of the Model.
  parent_artifact_ids = set()
  for event in store.get_events_by_execution_ids(execution_ids):
    if event_lib.is_valid_input_event(event):
      parent_artifact_ids.add(event.artifact_id)

  # Get the type ids of the parent artifacts.
  type_ids = set()
  artifact_by_artifact_id = {}
  for artifact in store.get_artifacts_by_id(parent_artifact_ids):
    type_ids.add(artifact.type_id)
    artifact_by_artifact_id[artifact.id] = artifact

  # Find the ArtifactType associated with Examples.
  for artifact_type in store.get_artifact_types_by_id(type_ids):
    if artifact_type.name == ops_utils.EXAMPLES_TYPE_NAME:
      examples_type = artifact_type
      break
  else:
    return []

  mlmd_examples = []
  for artifact_id in parent_artifact_ids:
    artifact = artifact_by_artifact_id[artifact_id]
    if artifact.type_id == examples_type.id:
      mlmd_examples.append(artifact)

  if not mlmd_examples:
    return []

  # Return the sorted Examples.
  artifacts = artifact_utils.deserialize_artifacts(examples_type, mlmd_examples)
  return sorted(
      artifacts, key=lambda a: (a.mlmd_artifact.create_time_since_epoch, a.id)
  )


class TrainingRange(
    resolver_op.ResolverOp,
    canonical_name='tfx.TrainingRange',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """TrainingRange operator."""

  def apply(
      self, input_list: Sequence[types.Artifact]
  ) -> Sequence[types.Artifact]:
    """Returns the Examples artifacts used to train the Model."""
    if not input_list:
      return []

    model = _validate_input_list(input_list)

    examples = training_range(self.context.store, model)

    if not examples:
      return []

    return examples
