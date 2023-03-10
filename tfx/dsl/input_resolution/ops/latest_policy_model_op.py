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
"""Module for LatestPolicyModel operator."""
import collections
import enum

from typing import Dict

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.mlmd import event_lib
from tfx.types import artifact_utils
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


# TODO(kshivvy): Consider supporting LATEST_PUSHED_DIFFERENT_DATA and
# LATEST_EVALUATOR_BLESSED_DIFFERENT_DATA for Evaluator input resolution.
class Policy(enum.IntEnum):
  """Policy for selecting the latest model."""

  # Get the latest trained Model.
  LATEST_EXPORTED = 1

  # Get the latest Model blessed by a Evaluator.
  LATEST_EVALUATOR_BLESSED = 2

  # Get the latest Model blessed by a InfraValidator.
  LATEST_INFRA_VALIDATOR_BLESSED = 3

  # Get the latest Model blessed by both a Evaluator and an InfraValidator.
  LATEST_BLESSED = 4

  # Get the latest Model pushed by a Pusher.
  LATEST_PUSHED = 5


class ModelRelations:
  """Stores child ModelBlessing, ModelInfraBlessing, ModelPush for a Model."""

  model_blessing_by_artifact_id: Dict[int, types.Artifact]
  infra_blessing_by_artifact_id: Dict[int, types.Artifact]
  model_push_by_artifact_id: Dict[int, types.Artifact]

  def __init__(self):
    self.model_blessing_by_artifact_id = {}
    self.infra_blessing_by_artifact_id = {}
    self.model_push_by_artifact_id = {}

  def meets_policy(self, policy: Policy) -> bool:
    """Checks if ModelRelations contains artifacts that meet the Policy."""
    if policy == Policy.LATEST_EXPORTED:
      return True
    elif policy == Policy.LATEST_PUSHED:
      return bool(self.model_push_by_artifact_id)
    elif policy == Policy.LATEST_EVALUATOR_BLESSED:
      return bool(self.model_blessing_by_artifact_id)
    elif policy == Policy.LATEST_INFRA_VALIDATOR_BLESSED:
      return bool(self.infra_blessing_by_artifact_id)
    elif policy == Policy.LATEST_BLESSED:
      return bool(self.model_blessing_by_artifact_id) and bool(
          self.infra_blessing_by_artifact_id
      )

    return False

  def latest_created(
      self, artifact_type: metadata_store_pb2.ArtifactType
  ) -> types.Artifact:
    """Gets the latest created artifact with matching ArtifactType."""
    if artifact_type.name == ops_utils.MODEL_BLESSING_TYPE_NAME:
      artifacts = self.model_blessing_by_artifact_id.values()
    elif artifact_type.name == ops_utils.MODEL_INFRA_BLESSSING_TYPE_NAME:
      artifacts = self.infra_blessing_by_artifact_id.values()
    elif artifact_type.name == ops_utils.MODEL_PUSH_TYPE_NAME:
      artifacts = self.model_push_by_artifact_id.values()
    else:
      raise exceptions.InvalidArgument(
          'ModelRelations.latest_created() can only be called with an '
          'artifact_type.name in '
          f'{ops_utils.MODEL_BLESSSING_KEY, ops_utils.MODEL_INFRA_BLESSING_KEY, ops_utils.MODEL_PUSH_KEY}.'
      )

    # Sort the artifacts by latest created, ties broken by id.
    sorted_artifacts = sorted(
        artifacts, key=lambda a: (a.create_time_since_epoch, a.id), reverse=True
    )

    # Deserialize the MLMD artifact to a TFleX Artifact object.
    return artifact_utils.deserialize_artifact(
        artifact_type, sorted_artifacts[0]
    )


def _is_eval_blessed(
    type_name: str, artifact: metadata_store_pb2.Artifact
) -> bool:
  """Checks if an MLMD artifact is a blessed ModelBlessing."""
  return (
      type_name == ops_utils.MODEL_BLESSING_TYPE_NAME
      and artifact.custom_properties['blessed'].int_value == 1
  )


def _is_infra_blessed(
    type_name: str, artifact: metadata_store_pb2.Artifact
) -> bool:
  """Checks if an MLMD artifact is a blessed ModelInfrablessing."""
  return (
      type_name == ops_utils.MODEL_INFRA_BLESSSING_TYPE_NAME
      and artifact.custom_properties['blessing_status'].string_value
      in ['INFRA_BLESSED', 'INFRA_DELEGATED']
  )


def _validate_input_dict(input_dict: typing_utils.ArtifactMultiMap):
  """Checks that the input_dict is properly formatted."""
  if ops_utils.MODEL_KEY not in input_dict.keys():
    raise exceptions.InvalidArgument(
        f'{ops_utils.MODEL_KEY} is a required key of the input_dict.'
    )

  valid_keys = {
      ops_utils.MODEL_KEY,
      ops_utils.MODEL_BLESSSING_KEY,
      ops_utils.MODEL_INFRA_BLESSING_KEY,
  }
  ops_utils.validate_input_dict(input_dict, valid_keys)


def _build_result_dictionary(
    result: typing_utils.ArtifactMultiDict,
    model_relations: ModelRelations,
    policy: int,
    artifact_type_by_name: Dict[str, metadata_store_pb2.ArtifactType],
):
  """Adds child artifacts to the result dictionary, based on the Policy."""
  if (
      policy == Policy.LATEST_EVALUATOR_BLESSED
      or policy == Policy.LATEST_BLESSED
  ):
    result[ops_utils.MODEL_BLESSSING_KEY] = [
        model_relations.latest_created(
            artifact_type_by_name[ops_utils.MODEL_BLESSING_TYPE_NAME]
        )
    ]

  # Intentionally use if instead of elif to handle LATEST_BLESSED Policy.
  if (
      policy == Policy.LATEST_INFRA_VALIDATOR_BLESSED
      or policy == Policy.LATEST_BLESSED
  ):
    result[ops_utils.MODEL_INFRA_BLESSING_KEY] = [
        model_relations.latest_created(
            artifact_type_by_name[ops_utils.MODEL_INFRA_BLESSSING_TYPE_NAME]
        )
    ]

  elif policy == Policy.LATEST_PUSHED:
    result[ops_utils.MODEL_PUSH_KEY] = [
        model_relations.latest_created(
            artifact_type_by_name[ops_utils.MODEL_PUSH_TYPE_NAME]
        )
    ]

  return result


class LatestPolicyModel(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestPolicyModel',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """LatestPolicyModel operator."""

  # The policy to select the model by. See Policy enum for valid options.
  # TODO(b/270621886): Restore to Policy Enum type.
  policy = resolver_op.Property(type=int)

  # If true, a SkipSignal will be raised. Else, an empty dictionary will be
  # returned. See Raises section below for full conditions in which a
  # SkipSignal raised/empty dict returned.
  raise_skip_signal = resolver_op.Property(type=bool, default=True)

  def _raise_skip_signal_or_return_empty_dict(self, error_msg: str = ''):
    if self.raise_skip_signal:
      raise exceptions.SkipSignal(error_msg)

    # Return a dictionary with the proper keys but empty artifact lists for
    # values.
    result = {ops_utils.MODEL_KEY: []}

    if (
        self.policy == Policy.LATEST_EVALUATOR_BLESSED
        or self.policy == Policy.LATEST_BLESSED
    ):
      result[ops_utils.MODEL_BLESSSING_KEY] = []

    if (
        self.policy == Policy.LATEST_INFRA_VALIDATOR_BLESSED
        or self.policy == Policy.LATEST_BLESSED
    ):
      result[ops_utils.MODEL_INFRA_BLESSING_KEY] = []

    elif self.policy == Policy.LATEST_PUSHED:
      result[ops_utils.MODEL_PUSH_KEY] = []

    return result

  def apply(self, input_dict: typing_utils.ArtifactMultiMap):
    """Finds the latest created model via a certain policy.

    The input_dict is expected to have the following format:

    {
        "model": [Model 1, Model 2, ...],
        "model_blessing": [ModelBlessing 1, ModelBlessing 2, ...],
        "model_infra_blessing": [ModelInfraBlessing 1, ...]
    }

    "model" is a required key. "model_blessing" and "model_infra_blessing" are
    optional keys. If "model_blessing" and/or "model_infra_blessing" are
    provided, then only their lineage w.r.t. the Model artifacts will be
    considered.

    Example usecases for specifying "model_blessing"/"model_infra_blessing"
    include: 1) Resolving inputs to a Pusher 2) Specifying ModelBlessing
    artifacts from a specific Evaluator, in cases where the pipeline has
    multiple Evaluators.

    Note that only the standard TFleX Model, ModelBlessing, ModelInfraBlessing,
    and ModelPush artifacts are supported.

    Args:
      input_dict: An input dict containing "model", "model_blessing",
        "model_infra_blessing" as keys and lists of Model, ModelBlessing, and
        ModelInfraBlessing artifacts as values, respectively.

    Returns:
      A dictionary containing the latest Model artifact, as well as the
      ModelBlessing, ModelInfraBlessing, and/or ModelPush based on the Policy.

      For example, for a LATEST_BLESSED policy, the following dict will be
      returned:
      {
        "model": [Model],
        "model_blessing": [ModelBlessing],
        "model_infra_blessing": [ModelInfraBlessing]
      }

      For a LATEST_PUSHED policy, the following dict will be returned:
      {
        "model": [Model],
        "model_push": [ModelPush]
      }

    Raises:
      InvalidArgument: If the models are not Model artifacts.
      SkipSignal: If raise_skip_signal is True and one of the following:
        1. The input_dict is empty.
        2. If no models are passed in.
        3. If input_dict contains "model_blessing" and/or "model_infra_blessing"
           as keys but have empty lists as values for both of them.
        4. No latest model was found that matches the policy.
    """
    if not input_dict:
      return self._raise_skip_signal_or_return_empty_dict(
          'The input dictionary is empty.'
      )

    _validate_input_dict(input_dict)

    if not input_dict[ops_utils.MODEL_KEY]:
      return self._raise_skip_signal_or_return_empty_dict(
          'The "model" key in the input dict contained no Model artifacts.'
      )

    # Sort the models from from latest created to oldest.
    models = input_dict.get(ops_utils.MODEL_KEY)
    models.sort(  # pytype: disable=attribute-error
        key=lambda a: (a.mlmd_artifact.create_time_since_epoch, a.id),
        reverse=True,
    )

    # Return the latest trained model if the policy is LATEST_EXPORTED.
    if self.policy == Policy.LATEST_EXPORTED:
      return {ops_utils.MODEL_KEY: [models[0]]}

    # If ModelBlessing and/or ModelInfraBlessing artifacts were included in
    # input_dict, then we will only consider those child artifacts.
    specifies_child_artifacts = (
        ops_utils.MODEL_BLESSSING_KEY in input_dict.keys()
        or ops_utils.MODEL_INFRA_BLESSING_KEY in input_dict.keys()
    )
    input_child_artifacts = input_dict.get(
        ops_utils.MODEL_BLESSSING_KEY, []
    ) + input_dict.get(ops_utils.MODEL_INFRA_BLESSING_KEY, [])
    input_child_artifact_ids = set([a.id for a in input_child_artifacts])

    # If the ModelBlessing and ModelInfraBlessing lists are empty, then no
    # child artifacts can be considered and we raise a SkipSignal. This can
    # occur when a Model has been trained but not blessed yet, for example.
    if specifies_child_artifacts and not input_child_artifact_ids:
      return self._raise_skip_signal_or_return_empty_dict(
          '"model_blessing" and/or "model_infra_blessing" were specified as '
          'keys in the input dictionary, but contained no '
          'ModelBlessing/ModelInfraBlessing artifacts.'
      )

    # In MLMD, two artifacts are related by:
    #
    #       Event 1           Event 2
    # Model ------> Execution ------> Artifact B
    #
    # Artifact B can be:
    # 1. ModelBlessing output artifact from an Evaluator.
    # 2. ModelInfraBlessing output artifact from an InfraValidator.
    # 3. ModelPush output artifact from a Pusher.
    #
    # We query MLMD to get a list of candidate model artifact ids that have
    # a child artifact of type child_artifact_type. Note we perform batch
    # queries to reduce the number round trips to the database.

    # There could be multiple events with the same execution ID but different
    # artifact IDs (e.g. model and baseline_model passed to an Evaluator), so we
    # keep the values of model_artifact_ids_by_execution_id as sets.
    model_artifact_ids = sorted(set(m.id for m in models))
    model_artifact_ids_by_execution_id = collections.defaultdict(set)

    # Pusher takes uses the key "model_export" to take in the Model artifact,
    # but all other components use the key "model".
    if self.policy == Policy.LATEST_PUSHED:
      event_input_key = ops_utils.MODEL_EXPORT_KEY
    else:
      event_input_key = ops_utils.MODEL_KEY

    # Get all Executions in MLMD associated with the Model artifacts.
    execution_ids = set()
    for event in self.context.store.get_events_by_artifact_ids(
        model_artifact_ids
    ):
      if event_lib.is_valid_input_event(event, event_input_key):
        model_artifact_ids_by_execution_id[event.execution_id].add(
            event.artifact_id
        )
        execution_ids.add(event.execution_id)

    # Get all artifact ids associated with an OUTPUT Event in each Execution.
    # These ids correspond to descendant artifacts 1 hop distance away from the
    # Model.
    child_artifact_ids = set()
    child_artifact_ids_by_model_artifact_id = collections.defaultdict(set)
    model_artifact_ids_by_child_artifact_id = collections.defaultdict(set)
    for event in self.context.store.get_events_by_execution_ids(execution_ids):
      if not event_lib.is_valid_output_event(event):
        continue

      child_artifact_id = event.artifact_id
      # Only consider child artifacts present in input_dict, if specified.
      if (
          specifies_child_artifacts
          and child_artifact_id not in input_child_artifact_ids
      ):
        continue

      child_artifact_ids.add(child_artifact_id)
      model_artifact_ids = model_artifact_ids_by_execution_id[
          event.execution_id
      ]
      model_artifact_ids_by_child_artifact_id[child_artifact_id] = (
          model_artifact_ids
      )

      for model_artifact_id in model_artifact_ids:
        child_artifact_ids_by_model_artifact_id[model_artifact_id].add(
            child_artifact_id
        )

    # Get Model, ModelBlessing, ModelInfraBlessing, and ModelPush ArtifactTypes.
    artifact_type_by_type_id = {}
    artifact_type_by_name = {}
    for artifact_type in self.context.store.get_artifact_types():
      artifact_type_by_type_id[artifact_type.id] = artifact_type
      artifact_type_by_name[artifact_type.name] = artifact_type

    # Populate the ModelRelations associated with each Model artifact and its
    # children.
    child_artifact_by_artifact_id = {}
    model_relations_by_model_artifact_id = collections.defaultdict(
        ModelRelations
    )
    for artifact in self.context.store.get_artifacts_by_id(child_artifact_ids):
      child_artifact_by_artifact_id[artifact.id] = artifact
      for model_artifact_id in model_artifact_ids_by_child_artifact_id[
          artifact.id
      ]:
        model_relations = model_relations_by_model_artifact_id[
            model_artifact_id
        ]

        artifact_type_name = artifact_type_by_type_id[artifact.type_id].name
        if _is_eval_blessed(artifact_type_name, artifact):
          model_relations.model_blessing_by_artifact_id[artifact.id] = artifact

        elif _is_infra_blessed(artifact_type_name, artifact):
          model_relations.infra_blessing_by_artifact_id[artifact.id] = artifact

        elif artifact_type_name == ops_utils.MODEL_PUSH_TYPE_NAME:
          model_relations.model_push_by_artifact_id[artifact.id] = artifact

    # Find the latest model and ModelRelations that meets the Policy.
    result = {}
    for model in models:
      model_relations = model_relations_by_model_artifact_id[model.id]
      if model_relations.meets_policy(self.policy):
        result[ops_utils.MODEL_KEY] = [model]
        break
    else:
      return self._raise_skip_signal_or_return_empty_dict(
          f'No model found that meets the Policy {Policy(self.policy).name}'
      )

    return _build_result_dictionary(
        result, model_relations, self.policy, artifact_type_by_name
    )
