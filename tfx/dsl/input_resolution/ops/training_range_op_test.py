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
"""Tests for tfx.dsl.input_resolution.ops.training_range_op."""

from typing import Dict, List, Optional

from absl.testing import parameterized

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.types import artifact_utils
from tfx.utils import test_case_utils as mlmd_mixins


class TrainingRangeOpTest(
    tf.test.TestCase, parameterized.TestCase, mlmd_mixins.MlmdMixins
):

  def _training_range(self, model: types.Artifact):
    return test_utils.run_resolver_op(
        ops.TrainingRange,
        [model],
        context=resolver_op.Context(store=self.store),
    )

  def _prepare_tfx_artifact(
      self,
      artifact: types.Artifact,
      properties: Optional[Dict[str, int | str]] = None,
  ) -> types.Artifact:
    """Adds a single artifact to MLMD and returns the TFleX Artifact object."""
    mlmd_artifact = self.put_artifact(artifact.TYPE_NAME, properties=properties)
    artifact_type = self.store.get_artifact_type(artifact.TYPE_NAME)
    return artifact_utils.deserialize_artifact(artifact_type, mlmd_artifact)

  def _unwrap_tfx_artifact(self, artifact: types.Artifact) -> types.Artifact:
    """Return the underlying MLMD Artifact of a TFleX Artifact object."""
    return artifact.mlmd_artifact

  def _train_on_examples(
      self, model: types.Artifact, examples: List[types.Artifact]
  ):
    """Add an Execution to MLMD where a Trainer trains on the examples."""
    self.put_execution(
        'TFTrainer',
        inputs={
            'examples': [self._unwrap_tfx_artifact(e) for e in examples],
            'transform_graph': [
                self._unwrap_tfx_artifact(self.transform_graph)
            ],
        },
        outputs={'model': [self._unwrap_tfx_artifact(model)]},
    )

  def _build_examples(self, n: int) -> List[types.Artifact]:
    return [
        self._prepare_tfx_artifact(
            test_utils.Examples, properties={'span': i, 'version': 1}
        )
        for i in range(n)
    ]

  def assertArtifactListEqual(
      self,
      actual: List[types.Artifact],
      expected: List[types.Artifact],
  ):
    self.assertEqual(len(actual), len(expected))
    for a, e in zip(actual, expected):
      self.assertEqual(str(a), str(e))
      self.assertEqual(a.mlmd_artifact, e.mlmd_artifact)

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.model = self._prepare_tfx_artifact(test_utils.Model)
    self.transform_graph = self._prepare_tfx_artifact(test_utils.TransformGraph)
    self.examples = self._build_examples(10)

  def testTrainingRangeOp_SingleModelExecution(self):
    self._train_on_examples(self.model, self.examples[:5])

    actual = self._training_range(self.model)
    self.assertArtifactListEqual(actual, self.examples[:5])

  def testTrainingRangeOp_MultipleModels(self):
    model_1 = self._prepare_tfx_artifact(test_utils.Model)
    self._train_on_examples(model_1, self.examples[:5])

    model_2 = self._prepare_tfx_artifact(test_utils.Model)
    self._train_on_examples(model_2, self.examples[5:])

    actual_1 = self._training_range(model_1)
    self.assertArtifactListEqual(actual_1, self.examples[:5])

    actual_2 = self._training_range(model_2)
    self.assertArtifactListEqual(actual_2, self.examples[5:])

  def testTrainingRangeOp_TrainOnTransformedExamples_ReturnsTransformedExamples(
      self,
  ):
    transformed_examples = self._build_examples(10)
    self.put_execution(
        'Transform',
        inputs={
            'examples': [self._unwrap_tfx_artifact(e) for e in self.examples],
        },
        outputs={
            'transformed_examples': [
                self._unwrap_tfx_artifact(e) for e in transformed_examples
            ],
            'transform_graph': [
                self._unwrap_tfx_artifact(self.transform_graph)
            ],
        },
    )

    self._train_on_examples(self.model, transformed_examples)
    actual = self._training_range(self.model)
    self.assertArtifactListEqual(actual, transformed_examples)

  def testTrainingRangeOp_SameSpanMultipleVersions_AllVersionsReturned(self):
    examples = [
        self._prepare_tfx_artifact(
            test_utils.Examples, properties={'span': 1, 'version': i}
        )
        for i in range(10)
    ]
    self._train_on_examples(self.model, examples[:5])

    actual = self._training_range(self.model)
    self.assertArtifactListEqual(actual, examples[:5])

  def testTrainingRangeOp_EmptyListReturned(self):
    # No input model artifact.
    actual = test_utils.run_resolver_op(
        ops.TrainingRange,
        [],
        context=resolver_op.Context(store=self.store),
    )
    self.assertEmpty(actual)

    # No executions in MLMD.
    actual = self._training_range(self.model)
    self.assertEmpty(actual)

    # Execution in MLMD does not have input Example artifacts.
    self.put_execution(
        'NoInputExamplesTFTrainer',
        inputs={},
        outputs={'model': [self._unwrap_tfx_artifact(self.model)]},
    )
    actual = self._training_range(self.model)
    self.assertEmpty(actual)

  def testTrainingRangeOp_InvalidArgumentRaised(self):
    with self.assertRaises(exceptions.InvalidArgument):
      # Two model artifacts.
      test_utils.run_resolver_op(
          ops.TrainingRange,
          [self.model, self.model],
          context=resolver_op.Context(store=self.store),
      )

      # Incorret input artifact type.
      test_utils.run_resolver_op(
          ops.TrainingRange,
          [self.transform_graph],
          context=resolver_op.Context(store=self.store),
      )

  def testTrainingRangeOp_BulkInferrerProducesExamples(self):
    self._train_on_examples(self.model, self.examples[:5])
    bulk_inferrer_examples = self._build_examples(5)

    # The BulkInferrer takes in the same Examples used to Trainer the Model,
    # and outputs 5 new examples to be used downstream. This creates additional
    # Examples artifacts in MLMD linked to the Model, but they should NOT be
    # returend as the Examples that the Model was trained on.
    self.put_execution(
        'TFTrainer',
        inputs={
            'examples': [self._unwrap_tfx_artifact(e) for e in self.examples],
            'model': [self._unwrap_tfx_artifact(self.model)],
        },
        outputs={
            'output_examples': [
                self._unwrap_tfx_artifact(e) for e in bulk_inferrer_examples
            ]
        },
    )

    actual = self._training_range(self.model)
    self.assertArtifactListEqual(actual, self.examples[:5])


if __name__ == '__main__':
  tf.test.main()
