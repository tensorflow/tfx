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
"""Tests for tfx.dsl.input_resolution.ops.latest_policy_model_op."""
from typing import Dict, List, Optional, Union

from absl.testing import parameterized

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import latest_policy_model_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.types import artifact_utils
from tfx.utils import test_case_utils as mlmd_mixins


_LATEST_EXPORTED = latest_policy_model_op.Policy.LATEST_EXPORTED
_LATEST_EVALUATOR_BLESSED = (
    latest_policy_model_op.Policy.LATEST_EVALUATOR_BLESSED
)
_LATEST_INFRA_VALIDATOR_BLESSED = (
    latest_policy_model_op.Policy.LATEST_INFRA_VALIDATOR_BLESSED
)
_LATEST_BLESSED = latest_policy_model_op.Policy.LATEST_BLESSED
_LATEST_PUSHED = latest_policy_model_op.Policy.LATEST_PUSHED


class _Model(test_utils.DummyArtifact):
  TYPE_NAME = latest_policy_model_op.MODEL_TYPE_NAME


class _ModelBlessing(test_utils.DummyArtifact):
  TYPE_NAME = latest_policy_model_op.MODEL_BLESSING_TYPE_NAME


class _ModelInfraBlessing(test_utils.DummyArtifact):
  TYPE_NAME = latest_policy_model_op.MODEL_INFRA_BLESSSING_TYPE_NAME


class _ModelPush(test_utils.DummyArtifact):
  TYPE_NAME = latest_policy_model_op.MODEL_PUSH_TYPE_NAME


class LatestPolicyModelOpTest(
    tf.test.TestCase, parameterized.TestCase, mlmd_mixins.MlmdMixins
):

  def _latest_policy_model(
      self,
      policy: latest_policy_model_op.Policy,
      model: Optional[List[types.Artifact]] = None,
      model_blessing: Optional[List[types.Artifact]] = None,
      model_infra_blessing: Optional[List[types.Artifact]] = None,
  ):
    """Run the LatestPolicyModel ResolverOp."""
    input_dict = {'model': model or self.artifacts}

    if model_blessing is not None:
      input_dict['model_blessing'] = model_blessing

    if model_infra_blessing is not None:
      input_dict['model_infra_blessing'] = model_infra_blessing

    return test_utils.run_resolver_op(
        ops.LatestPolicyModel,
        input_dict,
        context=resolver_op.Context(store=self.store),
        policy=policy,
    )

  def _prepare_tfx_artifact(
      self,
      artifact: types.Artifact,
      custom_properties: Optional[Dict[str, Union[int, str]]] = None,
  ) -> types.Artifact:
    """Adds a single artifact to MLMD and returns the TFleX Artifact object."""
    mlmd_artifact = self.put_artifact(
        artifact.TYPE_NAME, custom_properties=custom_properties
    )
    artifact_type = self.store.get_artifact_type(artifact.TYPE_NAME)
    return artifact_utils.deserialize_artifact(artifact_type, mlmd_artifact)

  def _unwrap_tfx_artifact(self, artifact: types.Artifact):
    """Return the underlying MLMD Artifact of a TFleX Artifact object."""
    return [artifact.mlmd_artifact]

  def _evaluator_bless_model(
      self,
      model: types.Artifact,
      blessed: bool = True,
      baseline_model: Optional[types.Artifact] = None,
  ) -> types.Artifact:
    """Add an Execution to MLMD where the Evaluator blesses the model."""
    if blessed:
      custom_properties = {'blessed': 1}
    else:
      custom_properties = {'blessed': 0}
    model_blessing = self._prepare_tfx_artifact(
        _ModelBlessing, custom_properties
    )

    inputs = {'model': self._unwrap_tfx_artifact(model)}
    if baseline_model is not None:
      inputs['baseline_model'] = self._unwrap_tfx_artifact(baseline_model)

    self.put_execution(
        'Evaluator',
        inputs=inputs,
        outputs={'blessing': self._unwrap_tfx_artifact(model_blessing)},
    )

    return model_blessing

  def _infra_validator_bless_model(
      self, model: types.Artifact, blessed: bool = True
  ) -> types.Artifact:
    """Add an Execution to MLMD where the InfraValidator blesses the model."""
    if blessed:
      custom_properties = {'blessing_status': 'INFRA_BLESSED'}
    else:
      custom_properties = {'blessing_status': 'INFRA_NOT_BLESSED'}
    model_infra_blessing = self._prepare_tfx_artifact(
        _ModelInfraBlessing, custom_properties
    )

    self.put_execution(
        'InfraValidator',
        inputs={'model': self._unwrap_tfx_artifact(model)},
        outputs={'result': self._unwrap_tfx_artifact(model_infra_blessing)},
    )

    return model_infra_blessing

  def _push_model(self, model: types.Artifact):
    """Add an Execution to MLMD where the Pusher pushes the model."""
    model_push = self._prepare_tfx_artifact(_ModelPush)
    self.put_execution(
        'ServomaticPusher',
        inputs={'model': self._unwrap_tfx_artifact(model)},
        outputs={'model_push': self._unwrap_tfx_artifact(model_push)},
    )
    return model_push

  def assertArtifactDictEqual(
      self,
      actual: Dict[str, types.Artifact],
      expected: Dict[str, types.Artifact],
  ):
    # The call to artifact_utils.deserialize_artifact() in
    # latest_policy_model_op._get_latest_created() results in an artifact with
    # the same string representation, but different object id (pointer in
    # memory), causing assertArtifactDictEqual to incorrectly fail.
    self.assertEqual(actual.keys(), expected.keys())
    for key in actual:
      self.assertEqual(str(actual[key]), str(expected[key]))
      self.assertEqual(actual[key].mlmd_artifact, expected[key].mlmd_artifact)

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.model_1 = self._prepare_tfx_artifact(_Model)
    self.model_2 = self._prepare_tfx_artifact(_Model)
    self.model_3 = self._prepare_tfx_artifact(_Model)

    self.artifacts = [self.model_1, self.model_2, self.model_3]

  def testLatestPolicyModelOpTest_EmptyInput_RaiesSkipSignal(self):
    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_op(
          ops.LatestPolicyModel,
          {},
          policy=latest_policy_model_op.Policy.LATEST_EXPORTED,
          context=resolver_op.Context(store=self.store),
      )

  def testLatestPolicyModelOpTest_ValidateInputDict(self):
    with self.assertRaises(exceptions.SkipSignal):
      # Empty input dictionary.
      latest_policy_model_op._validate_input_dict({})

      # 'model' key present but contains no artifacts.
      input_dict = {'model_blessing': []}
      latest_policy_model_op._validate_input_dict(input_dict)

    with self.assertRaises(exceptions.InvalidArgument):
      # "model" key is missing.
      input_dict = {'model_blessing': [self.model_1]}
      latest_policy_model_op._validate_input_dict(input_dict)

      # Invalid key "foo".
      input_dict = {'model': [self.model_1], 'foo': [self.model_1]}
      latest_policy_model_op._validate_input_dict(input_dict)

      # Incorrect artifact type for "model_infra_blessing".
      input_dict = {
          'model': [self.model_1],
          'model_infra_blessing': [self.model_1],
      }
      latest_policy_model_op._validate_input_dict(input_dict)

      # E2E call results in InvalidArgument.
      self._latest_policy_model(
          _LATEST_EVALUATOR_BLESSED,
          model=[self.model_1],
          model_blessing=[self.model_1],
      )

    model_infra_blessing = self._infra_validator_bless_model(self.model_1)
    model_blessing = self._evaluator_bless_model(self.model_1)

    # Should not raise any exception.
    input_dict = {
        'model': [self.model_1],
        'model_blessing': [model_blessing],
        'model_infra_blessing': [model_infra_blessing],
    }
    latest_policy_model_op._validate_input_dict(input_dict)

  def testLatestPolicyModelOpTest_LatestTrainedModel(self):
    actual = self._latest_policy_model(_LATEST_EXPORTED)
    self.assertArtifactDictEqual(actual, {'model': self.model_3})

  def testLatestPolicyModelOp_SeqeuntialExecutions_LatestModelChanges(self):
    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
      self._latest_policy_model(_LATEST_BLESSED)

    # Insert spurious Executions.
    self._push_model(self.model_1)
    infra_blessing_2 = self._infra_validator_bless_model(self.model_2)
    model_push_3 = self._push_model(self.model_3)

    model_blessing_1 = self._evaluator_bless_model(self.model_1)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_1, 'model_blessing': model_blessing_1}
    )

    model_blessing_3 = self._evaluator_bless_model(self.model_3)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_3, 'model_blessing': model_blessing_3}
    )

    # No model has been blessed by both the Evaluator and InfraValidator yet.
    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_BLESSED)

    # model_3 should still be the latest Evaluator blessed model, since it is
    # the latest created.
    model_blessing_2 = self._evaluator_bless_model(self.model_2)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_3, 'model_blessing': model_blessing_3}
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {
            'model': self.model_2,
            'model_blessing': model_blessing_2,
            'model_infra_blessing': infra_blessing_2,
        },
    )

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_3, 'model_push': model_push_3}
    )

  def testLatestPolicyModelOp_NonBlessedArtifacts(self):
    self._infra_validator_bless_model(self.model_1, blessed=False)
    self._infra_validator_bless_model(self.model_2, blessed=False)
    self._infra_validator_bless_model(self.model_3, blessed=False)

    self._evaluator_bless_model(self.model_1, blessed=False)
    self._evaluator_bless_model(self.model_2, blessed=False)
    self._evaluator_bless_model(self.model_3, blessed=False)

    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
      self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
      self._latest_policy_model(_LATEST_BLESSED)
      self._latest_policy_model(_LATEST_PUSHED)

    model_push_1 = self._push_model(self.model_1)

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_1, 'model_push': model_push_1}
    )

    model_blessing_1 = self._evaluator_bless_model(self.model_1, blessed=True)
    model_infra_blessing_2 = self._infra_validator_bless_model(
        self.model_2, blessed=True
    )

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_1, 'model_blessing': model_blessing_1}
    )

    actual = self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {'model': self.model_2, 'model_infra_blessing': model_infra_blessing_2},
    )

    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_BLESSED)

    model_blessing_2 = self._evaluator_bless_model(self.model_2, blessed=True)

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2}
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {
            'model': self.model_2,
            'model_infra_blessing': model_infra_blessing_2,
            'model_blessing': model_blessing_2,
        },
    )

  def testLatestPolicyModelOp_VaryingPolicy(self):
    model_push = self._push_model(self.model_3)
    model_infra_blessing_1 = self._infra_validator_bless_model(self.model_1)
    model_infra_blessing_2 = self._infra_validator_bless_model(self.model_2)

    # Evaluator blessses Model 1 twice.
    self._evaluator_bless_model(self.model_1)
    model_blessing_1_2 = self._evaluator_bless_model(self.model_1)

    actual = self._latest_policy_model(_LATEST_EXPORTED)
    self.assertArtifactDictEqual(actual, {'model': self.model_3})

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {'model': self.model_1, 'model_blessing': model_blessing_1_2},
    )

    actual = self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {'model': self.model_2, 'model_infra_blessing': model_infra_blessing_2},
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactDictEqual(
        actual,
        {
            'model': self.model_1,
            'model_blessing': model_blessing_1_2,
            'model_infra_blessing': model_infra_blessing_1,
        },
    )

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_3, 'model_push': model_push}
    )

  def testLatestPolicyModelOp_MultipleModelInputEventsSameExecutionId(self):
    model_blessing_2_1 = self._evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_1
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2_1}
    )

    # Bless Model 2 again, using the same baseline Model 1 as before.
    model_blessing_2_2 = self._evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_1
    )
    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED, model=[self.model_2, self.model_3]
    )
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2_2}
    )

    # Model 2 should be returned as the latest blessed model, even though
    # there exists an Event between Model 3 and a ModelBlessing. In practice
    # however, the baseline_model will be created earlier than the model.
    model_blessing_2_3 = self._evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_3
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2_3}
    )

    model_blessing_3 = self._evaluator_bless_model(
        model=self.model_3, baseline_model=self.model_2
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactDictEqual(
        actual, {'model': self.model_3, 'model_blessing': model_blessing_3}
    )

    # When we restrict the artifacts to just [Model 1, Model 2], then Model 2
    # should be returned.
    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED, model=[self.model_1, self.model_2]
    )
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2_3}
    )

  def testLatestPolicyModelOp_InputDictContainsAllKeys(self):
    model_blessing_1 = self._evaluator_bless_model(model=self.model_1)
    model_infra_blessing_1 = self._infra_validator_bless_model(
        model=self.model_1
    )
    model_blessing_2 = self._evaluator_bless_model(model=self.model_2)

    # Spurious blessings that will not be included in input_dict.
    model_infra_blessing_2 = self._infra_validator_bless_model(
        model=self.model_2
    )
    self._evaluator_bless_model(model=self.model_3)
    self._infra_validator_bless_model(model=self.model_3)

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1],
        model_infra_blessing=[],
    )
    self.assertArtifactDictEqual(
        actual, {'model': self.model_1, 'model_blessing': model_blessing_1}
    )

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[],
    )
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2}
    )

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1],
    )
    self.assertArtifactDictEqual(
        actual, {'model': self.model_2, 'model_blessing': model_blessing_2}
    )

    actual = self._latest_policy_model(
        _LATEST_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1, model_infra_blessing_2],
    )
    self.assertArtifactDictEqual(
        actual,
        {
            'model': self.model_2,
            'model_blessing': model_blessing_2,
            'model_infra_blessing': model_infra_blessing_2,
        },
    )

    actual = self._latest_policy_model(
        _LATEST_BLESSED,
        model=[self.model_1, self.model_3],
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1, model_infra_blessing_2],
    )
    self.assertArtifactDictEqual(
        actual,
        {
            'model': self.model_1,
            'model_blessing': model_blessing_1,
            'model_infra_blessing': model_infra_blessing_1,
        },
    )

  @parameterized.parameters(
      (['m1'], [], [], _LATEST_EVALUATOR_BLESSED, ['m1']),
      ([], ['m1'], [], _LATEST_INFRA_VALIDATOR_BLESSED, ['m1']),
      (['m1'], ['m1'], [], _LATEST_BLESSED, ['m1']),
      ([], [], ['m1'], _LATEST_PUSHED, ['m1']),
      (
          ['m1', 'm2', 'm3'],
          ['m2', 'm3'],
          ['m3'],
          _LATEST_EVALUATOR_BLESSED,
          ['m3'],
      ),
      (
          ['m1', 'm2', 'm3'],
          ['m2', 'm3'],
          ['m3'],
          _LATEST_INFRA_VALIDATOR_BLESSED,
          ['m3'],
      ),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], _LATEST_BLESSED, ['m3']),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], _LATEST_PUSHED, ['m3']),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m1'], _LATEST_PUSHED, ['m1']),
      (['m2', 'm1'], [], [], _LATEST_EVALUATOR_BLESSED, ['m2']),
  )
  def testLatestPolicyModelOp_RealisticModelExecutions_ModelResolvedCorrectly(
      self,
      eval_models: List[types.Artifact],
      infra_val_models: List[types.Artifact],
      push_models: List[types.Artifact],
      policy: latest_policy_model_op.Policy,
      expected: List[types.Artifact],
  ):
    str_to_model = {
        'm1': self.model_1,
        'm2': self.model_2,
        'm3': self.model_3,
    }

    for model in eval_models:
      self._evaluator_bless_model(str_to_model[model])

    for model in infra_val_models:
      self._infra_validator_bless_model(str_to_model[model])

    for model in push_models:
      self._push_model(str_to_model[model])

    actual = self._latest_policy_model(policy)['model']
    self.assertEqual(actual, str_to_model[expected[0]])


if __name__ == '__main__':
  tf.test.main()
