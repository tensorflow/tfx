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
from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2


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


def training_range(
    store: Any, model: types.Artifact, use_transformed_examples: bool = False
) -> List[types.Artifact]:
  """ContainsTrainingRange implementation, for shared use across ResolverOps.

  Returns the Examples artifact the Model was trained on.

  Note that only the standard TFleX Model and Examples artifacts are supported.

  Args:
   store: The MetadataStore.
   model: The Model artifact whose trained Examples to return.
   use_transformed_examples: Whether to return the materialized Examples
     produced by the Transform component. Should only be used if a Model was
     trained on materialized transformed Examples produced by a Transform.
     Defaults to False.

  Returns:
    List of Examples artifacts if found, else empty list. We intentionally don't
    raise SkipSignal, such that the caller can decide to raise it or not.
  """
  # In MLMD, an Examples and Model are related by:
  #
  #          Event 1           Event 2
  # Examples ------> Execution ------> Model
  #
  #
  # or, in the case where a Transform component materializes Examples:
  #
  #          Event 1           Event 2          Event 3
  # Examples ------> Execution ------> Examples -----> Execution ------> Model
  #
  #
  # For a single Model, there may be many parent Examples it was trained on.

  # TODO(kshivvy): Support querying multiple Model ids at once, to reduce the
  # number of round trip MLMD queries. This will be useful for resolving inputs
  # of a span driven evaluator.

  # Get all upstream Examples artifacts associated with the Model.
  mlmd_resolver = metadata_resolver.MetadataResolver(store)
  upstream_examples_dict = mlmd_resolver.get_upstream_artifacts_by_artifact_ids(
      artifact_ids=[model.id],
      # In MLMD, artifacts are 2 hops away. Because we are considering
      # Example -> (transformd) Examples -> Model, we set max_num_hops to 4.
      max_num_hops=4,
      filter_query=f'type="{ops_utils.EXAMPLES_TYPE_NAME}"',
  )
  if not upstream_examples_dict:
    return []
  upstream_example_and_type = upstream_examples_dict[model.id]
  if not upstream_example_and_type:
    return []

  # Get the sets of artifact IDs for Examples produced by Transform and by
  # ExampleGen.
  all_examples_ids = {a.id for a, _ in upstream_example_and_type}
  transformed_examples_ids = set()
  for event in store.get_events_by_artifact_ids(all_examples_ids):
    if event_lib.is_valid_output_event(
        event, expected_output_key=ops_utils.TRANSFORMED_EXAMPLES_KEY
    ):
      transformed_examples_ids.add(event.artifact_id)
  # We intentionally do set subtraction instead of filtering by the output_key
  # "examples", in case the Examples artifact is produced by a custom
  # component.
  examples_ids = all_examples_ids - transformed_examples_ids

  mlmd_artifacts = []
  for artifact, _ in upstream_example_and_type:
    # Only consider Examples artifacts that are marked LIVE. This excludes
    # garbage collected artifacts (which are marked as DELETED).
    if artifact.state != metadata_store_pb2.Artifact.State.LIVE:
      continue
    elif use_transformed_examples and artifact.id in transformed_examples_ids:
      mlmd_artifacts.append(artifact)
    elif not use_transformed_examples and artifact.id in examples_ids:
      mlmd_artifacts.append(artifact)
  if not mlmd_artifacts:
    return []

  # Find the ArtifactType associated with the artifacts.
  artifact_type_by_id = {t.id: t for _, t in upstream_example_and_type}
  artifact_type = artifact_type_by_id[mlmd_artifacts[0].type_id]

  # Return the sorted, serialized Examples.
  artifacts = artifact_utils.deserialize_artifacts(
      artifact_type, mlmd_artifacts
  )
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

  # Whether to return the materialized Examples produced by the Transform
  # component. Should only be used if a Model was trained on materialized
  # transformed Examples produced by a Transform. Defaults to False.
  use_transformed_examples = resolver_op.Property(type=bool, default=False)

  def apply(
      self, input_list: Sequence[types.Artifact]
  ) -> Sequence[types.Artifact]:
    """Returns the Examples artifacts used to train the Model."""
    if not input_list:
      return []

    model = _validate_input_list(input_list)

    examples = training_range(
        self.context.store, model, self.use_transformed_examples
    )

    if not examples:
      return []

    return examples
