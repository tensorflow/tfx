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
from typing import Dict, List, Optional

from absl.testing import parameterized

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import latest_policy_model_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions


_LATEST_EXPORTED = latest_policy_model_op.Policy.LATEST_EXPORTED
_LATEST_EVALUATOR_BLESSED = (
    latest_policy_model_op.Policy.LATEST_EVALUATOR_BLESSED
)
_LATEST_INFRA_VALIDATOR_BLESSED = (
    latest_policy_model_op.Policy.LATEST_INFRA_VALIDATOR_BLESSED
)
_LATEST_BLESSED = latest_policy_model_op.Policy.LATEST_BLESSED
_LATEST_PUSHED = latest_policy_model_op.Policy.LATEST_PUSHED


class LatestPolicyModelOpTest(
    test_utils.ResolverTestCase,
):

  def _latest_policy_model(
      self,
      policy: latest_policy_model_op.Policy,
      raise_skip_signal=True,
      model: Optional[List[types.Artifact]] = None,
      model_blessing: Optional[List[types.Artifact]] = None,
      model_infra_blessing: Optional[List[types.Artifact]] = None,
  ):
    """Run the LatestPolicyModel ResolverOp."""
    if model is None:
      input_dict = {'model': self.artifacts}
    else:
      input_dict = {'model': model}

    if model_blessing is not None:
      input_dict['model_blessing'] = model_blessing

    if model_infra_blessing is not None:
      input_dict['model_infra_blessing'] = model_infra_blessing

    return self._run_latest_policy_model(
        input_dict, policy=policy, raise_skip_signal=raise_skip_signal
    )

  def _run_latest_policy_model(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.LatestPolicyModel,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.model_1 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_2 = self.prepare_tfx_artifact(test_utils.Model)
    self.model_3 = self.prepare_tfx_artifact(test_utils.Model)

    self.artifacts = [self.model_1, self.model_2, self.model_3]

  def assertDictKeysEmpty(
      self,
      output_dict: Dict[str, List[types.Artifact]],
      policy: latest_policy_model_op.Policy,
  ):
    # Check that the corresponding Policy keys are in the output dictionary.
    self.assertIn('model', output_dict)
    if policy == _LATEST_EVALUATOR_BLESSED or policy == _LATEST_BLESSED:
      self.assertIn('model_blessing', output_dict)
    elif policy == _LATEST_INFRA_VALIDATOR_BLESSED or policy == _LATEST_BLESSED:
      self.assertIn('model_infra_blessing', output_dict)
    elif policy == _LATEST_PUSHED:
      self.assertIn('model', output_dict)

    # Check that all the artifact lists are empty.
    for artifacts in output_dict.values():
      self.assertEmpty(artifacts)

  def testLatestPolicyModelOpTest_RaisesSkipSignal(self):
    with self.assertRaises(exceptions.SkipSignal):
      test_utils.run_resolver_op(
          ops.LatestPolicyModel,
          {},
          policy=_LATEST_EXPORTED,
          raise_skip_signal=True,
          context=resolver_op.Context(store=self.store),
      )

      # Keys present in input_dict but contains no artifacts.
      self._latest_policy_model(_LATEST_EXPORTED, model=[])
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED, model_blessing=[])
      self._latest_policy_model(
          _LATEST_INFRA_VALIDATOR_BLESSED, model_infra_blessing=[]
      )
      self._latest_policy_model(
          _LATEST_BLESSED, model_blessing=[], model_infra_blessing=[]
      )

      # Models present in input_dict but none of them meet the specified policy.
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
      self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
      self._latest_policy_model(_LATEST_BLESSED)
      self._latest_policy_model(_LATEST_PUSHED)

  def testLatestPolicyModelOpTest_DoesNotRaiseSkipSignal(self):
    self.assertDictKeysEmpty(
        test_utils.run_resolver_op(
            ops.LatestPolicyModel,
            {},
            policy=_LATEST_EXPORTED,
            raise_skip_signal=False,
            context=resolver_op.Context(store=self.store),
        ),
        policy=_LATEST_EXPORTED,
    )

    # Keys present in input_dict but contains no artifacts.
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_EXPORTED, raise_skip_signal=False, model=[]
        ),
        policy=_LATEST_EXPORTED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_EVALUATOR_BLESSED,
            raise_skip_signal=False,
            model_blessing=[],
        ),
        policy=_LATEST_EXPORTED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_INFRA_VALIDATOR_BLESSED,
            raise_skip_signal=False,
            model_infra_blessing=[],
        ),
        policy=_LATEST_INFRA_VALIDATOR_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_BLESSED,
            raise_skip_signal=False,
            model_blessing=[],
            model_infra_blessing=[],
        ),
        policy=_LATEST_BLESSED,
    )

    # Models present in input_dict but none of them meet the specified policy.
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_EVALUATOR_BLESSED, raise_skip_signal=False
        ),
        policy=_LATEST_EVALUATOR_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_INFRA_VALIDATOR_BLESSED, raise_skip_signal=False
        ),
        policy=_LATEST_INFRA_VALIDATOR_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(_LATEST_BLESSED, raise_skip_signal=False),
        policy=_LATEST_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(_LATEST_PUSHED, raise_skip_signal=False),
        policy=_LATEST_PUSHED,
    )

  def testLatestPolicyModelOpTest_ValidateInputDict(self):
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

    model_infra_blessing = self.infra_validator_bless_model(self.model_1)
    model_blessing = self.evaluator_bless_model(self.model_1)

    # Should not raise any exception.
    input_dict = {
        'model': [self.model_1],
        'model_blessing': [model_blessing],
        'model_infra_blessing': [model_infra_blessing],
    }
    latest_policy_model_op._validate_input_dict(input_dict)

  def testLatestPolicyModelOpTest_LatestTrainedModel(self):
    actual = self._latest_policy_model(_LATEST_EXPORTED)
    self.assertArtifactMapsEqual(actual, {'model': [self.model_3]})

  def testLatestPolicyModelOp_SeqeuntialExecutions_LatestModelChanges(self):
    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
      self._latest_policy_model(_LATEST_BLESSED)

    # Insert spurious Executions.
    self.push_model(self.model_1)
    infra_blessing_2 = self.infra_validator_bless_model(self.model_2)
    model_push_3 = self.push_model(self.model_3)

    model_blessing_1 = self.evaluator_bless_model(self.model_1)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_1], 'model_blessing': [model_blessing_1]}
    )

    model_blessing_3 = self.evaluator_bless_model(self.model_3)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_3], 'model_blessing': [model_blessing_3]}
    )

    # No model has been blessed by both the Evaluator and InfraValidator yet.
    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_BLESSED)

    # model_3 should still be the latest Evaluator blessed model, since it is
    # the latest created.
    model_blessing_2 = self.evaluator_bless_model(self.model_2)
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_3], 'model_blessing': [model_blessing_3]}
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_2],
            'model_blessing': [model_blessing_2],
            'model_infra_blessing': [infra_blessing_2],
        },
    )

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_3], 'model_push': [model_push_3]}
    )

  def testLatestPolicyModelOp_NonBlessedArtifacts(self):
    self.infra_validator_bless_model(self.model_1, blessed=False)
    self.infra_validator_bless_model(self.model_2, blessed=False)
    self.infra_validator_bless_model(self.model_3, blessed=False)

    self.evaluator_bless_model(self.model_1, blessed=False)
    self.evaluator_bless_model(self.model_2, blessed=False)
    self.evaluator_bless_model(self.model_3, blessed=False)

    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
      self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
      self._latest_policy_model(_LATEST_BLESSED)
      self._latest_policy_model(_LATEST_PUSHED)

    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_EVALUATOR_BLESSED, raise_skip_signal=False
        ),
        policy=_LATEST_EVALUATOR_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(
            _LATEST_INFRA_VALIDATOR_BLESSED, raise_skip_signal=False
        ),
        policy=_LATEST_INFRA_VALIDATOR_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(_LATEST_BLESSED, raise_skip_signal=False),
        policy=_LATEST_BLESSED,
    )
    self.assertDictKeysEmpty(
        self._latest_policy_model(_LATEST_PUSHED, raise_skip_signal=False),
        policy=_LATEST_PUSHED,
    )

    model_push_1 = self.push_model(self.model_1)

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_1], 'model_push': [model_push_1]}
    )

    model_blessing_1 = self.evaluator_bless_model(self.model_1, blessed=True)
    model_infra_blessing_2 = self.infra_validator_bless_model(
        self.model_2, blessed=True
    )

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_1], 'model_blessing': [model_blessing_1]}
    )

    actual = self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_2],
            'model_infra_blessing': [model_infra_blessing_2],
        },
    )

    with self.assertRaises(exceptions.SkipSignal):
      self._latest_policy_model(_LATEST_BLESSED)

    model_blessing_2 = self.evaluator_bless_model(self.model_2, blessed=True)

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_2], 'model_blessing': [model_blessing_2]}
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_2],
            'model_infra_blessing': [model_infra_blessing_2],
            'model_blessing': [model_blessing_2],
        },
    )

  def testLatestPolicyModelOp_VaryingPolicy(self):
    model_push = self.push_model(self.model_3)
    model_infra_blessing_1 = self.infra_validator_bless_model(self.model_1)
    model_infra_blessing_2 = self.infra_validator_bless_model(self.model_2)

    # Evaluator blessses Model 1 twice.
    self.evaluator_bless_model(self.model_1)
    model_blessing_1_2 = self.evaluator_bless_model(self.model_1)

    actual = self._latest_policy_model(_LATEST_EXPORTED)
    self.assertArtifactMapsEqual(actual, {'model': [self.model_3]})

    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {'model': [self.model_1], 'model_blessing': [model_blessing_1_2]},
    )

    actual = self._latest_policy_model(_LATEST_INFRA_VALIDATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_2],
            'model_infra_blessing': [model_infra_blessing_2],
        },
    )

    actual = self._latest_policy_model(_LATEST_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_1],
            'model_blessing': [model_blessing_1_2],
            'model_infra_blessing': [model_infra_blessing_1],
        },
    )

    actual = self._latest_policy_model(_LATEST_PUSHED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_3], 'model_push': [model_push]}
    )

  def testLatestPolicyModelOp_MultipleModelInputEventsSameExecutionId(self):
    model_blessing_2_1 = self.evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_1
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {'model': [self.model_2], 'model_blessing': [model_blessing_2_1]},
    )

    # Bless Model 2 again, using the same baseline Model 1 as before.
    model_blessing_2_2 = self.evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_1
    )
    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED, model=[self.model_2, self.model_3]
    )
    self.assertArtifactMapsEqual(
        actual,
        {'model': [self.model_2], 'model_blessing': [model_blessing_2_2]},
    )

    # Model 2 should be returned as the latest blessed model, even though
    # there exists an Event between Model 3 and a ModelBlessing. In practice
    # however, the baseline_model will be created earlier than the model.
    model_blessing_2_3 = self.evaluator_bless_model(
        model=self.model_2, baseline_model=self.model_3
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual,
        {'model': [self.model_2], 'model_blessing': [model_blessing_2_3]},
    )

    model_blessing_3 = self.evaluator_bless_model(
        model=self.model_3, baseline_model=self.model_2
    )
    actual = self._latest_policy_model(_LATEST_EVALUATOR_BLESSED)
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_3], 'model_blessing': [model_blessing_3]}
    )

    # When we restrict the artifacts to just [Model 1, Model 2], then Model 2
    # should be returned.
    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED, model=[self.model_1, self.model_2]
    )
    self.assertArtifactMapsEqual(
        actual,
        {'model': [self.model_2], 'model_blessing': [model_blessing_2_3]},
    )

  def testLatestPolicyModelOp_InputDictContainsAllKeys(self):
    model_blessing_1 = self.evaluator_bless_model(model=self.model_1)
    model_infra_blessing_1 = self.infra_validator_bless_model(
        model=self.model_1
    )
    model_blessing_2 = self.evaluator_bless_model(model=self.model_2)

    # Spurious blessings that will not be included in input_dict.
    model_infra_blessing_2 = self.infra_validator_bless_model(
        model=self.model_2
    )
    self.evaluator_bless_model(model=self.model_3)
    self.infra_validator_bless_model(model=self.model_3)

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1],
        model_infra_blessing=[],
    )
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_1], 'model_blessing': [model_blessing_1]}
    )

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[],
    )
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_2], 'model_blessing': [model_blessing_2]}
    )

    actual = self._latest_policy_model(
        _LATEST_EVALUATOR_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1],
    )
    self.assertArtifactMapsEqual(
        actual, {'model': [self.model_2], 'model_blessing': [model_blessing_2]}
    )

    actual = self._latest_policy_model(
        _LATEST_BLESSED,
        model=self.artifacts,
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1, model_infra_blessing_2],
    )
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_2],
            'model_blessing': [model_blessing_2],
            'model_infra_blessing': [model_infra_blessing_2],
        },
    )

    actual = self._latest_policy_model(
        _LATEST_BLESSED,
        model=[self.model_1, self.model_3],
        model_blessing=[model_blessing_1, model_blessing_2],
        model_infra_blessing=[model_infra_blessing_1, model_infra_blessing_2],
    )
    self.assertArtifactMapsEqual(
        actual,
        {
            'model': [self.model_1],
            'model_blessing': [model_blessing_1],
            'model_infra_blessing': [model_infra_blessing_1],
        },
    )

  @parameterized.parameters(
      (['m1'], [], [], _LATEST_EVALUATOR_BLESSED, 'm1'),
      ([], ['m1'], [], _LATEST_INFRA_VALIDATOR_BLESSED, 'm1'),
      (['m1'], ['m1'], [], _LATEST_BLESSED, 'm1'),
      ([], [], ['m1'], _LATEST_PUSHED, 'm1'),
      (
          ['m1', 'm2', 'm3'],
          ['m2', 'm3'],
          ['m3'],
          _LATEST_EVALUATOR_BLESSED,
          'm3',
      ),
      (
          ['m1', 'm2', 'm3'],
          ['m2', 'm3'],
          ['m3'],
          _LATEST_INFRA_VALIDATOR_BLESSED,
          'm3',
      ),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], _LATEST_BLESSED, 'm3'),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m3'], _LATEST_PUSHED, 'm3'),
      (['m1', 'm2', 'm3'], ['m2', 'm3'], ['m1'], _LATEST_PUSHED, 'm1'),
      (['m2', 'm1'], [], [], _LATEST_EVALUATOR_BLESSED, 'm2'),
  )
  def testLatestPolicyModelOp_RealisticModelExecutions_ModelResolvedCorrectly(
      self,
      eval_models: List[str],
      infra_val_models: List[str],
      push_models: List[str],
      policy: latest_policy_model_op.Policy,
      expected: str,
  ):
    str_to_model = {
        'm1': self.model_1,
        'm2': self.model_2,
        'm3': self.model_3,
    }

    for model in eval_models:
      self.evaluator_bless_model(str_to_model[model])

    for model in infra_val_models:
      self.infra_validator_bless_model(str_to_model[model])

    for model in push_models:
      self.push_model(str_to_model[model])

    actual = self._latest_policy_model(policy)['model'][0]
    self.assertArtifactEqual(actual, str_to_model[expected])


if __name__ == '__main__':
  tf.test.main()
