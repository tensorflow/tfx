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

from typing import List

import tensorflow as tf

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration.portable.input_resolution import exceptions


class TrainingRangeOpTest(
    test_utils.ResolverTestCase,
):

  def _training_range(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.TrainingRange,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def _build_examples(self, n: int) -> List[types.Artifact]:
    return [
        self.prepare_tfx_artifact(
            test_utils.Examples, properties={'span': i, 'version': 1}
        )
        for i in range(n)
    ]

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.model = self.prepare_tfx_artifact(test_utils.Model)
    self.transform_graph = self.prepare_tfx_artifact(test_utils.TransformGraph)
    self.examples = self._build_examples(10)

  def testTrainingRangeOp_SingleModelExecution(self):
    self.train_on_examples(self.model, self.examples[:5], self.transform_graph)

    actual = self._training_range([self.model])
    self.assertArtifactListEqual(actual, self.examples[:5])

  def testTrainingRangeOp_MultipleModels(self):
    model_1 = self.prepare_tfx_artifact(test_utils.Model)
    self.train_on_examples(model_1, self.examples[:5], self.transform_graph)

    model_2 = self.prepare_tfx_artifact(test_utils.Model)
    self.train_on_examples(model_2, self.examples[5:], self.transform_graph)

    actual_1 = self._training_range([model_1])
    self.assertArtifactListEqual(actual_1, self.examples[:5])

    actual_2 = self._training_range([model_2])
    self.assertArtifactListEqual(actual_2, self.examples[5:])

  def testTrainingRangeOp_TrainOnTransformedExamples_ReturnsTransformedExamples(
      self,
  ):
    transformed_examples = self._build_examples(10)
    self.put_execution(
        'Transform',
        inputs={
            'examples': self.unwrap_tfx_artifacts(self.examples),
        },
        outputs={
            'transformed_examples': self.unwrap_tfx_artifacts(
                transformed_examples
            ),
            'transform_graph': self.unwrap_tfx_artifacts(
                [self.transform_graph]
            ),
        },
    )

    self.train_on_examples(
        self.model, transformed_examples, self.transform_graph
    )
    actual = self._training_range([self.model])
    self.assertArtifactListEqual(actual, transformed_examples)

  def testTrainingRangeOp_SameSpanMultipleVersions_AllVersionsReturned(self):
    examples = [
        self.prepare_tfx_artifact(
            test_utils.Examples, properties={'span': 1, 'version': i}
        )
        for i in range(10)
    ]
    self.train_on_examples(self.model, examples[:5], self.transform_graph)

    actual = self._training_range([self.model])
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
    actual = self._training_range([self.model])
    self.assertEmpty(actual)

    # Execution in MLMD does not have input Example artifacts.
    self.put_execution(
        'NoInputExamplesTFTrainer',
        inputs={},
        outputs={'model': self.unwrap_tfx_artifacts([self.model])},
    )
    actual = self._training_range([self.model])
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
    self.train_on_examples(self.model, self.examples[:5], self.transform_graph)
    bulk_inferrer_examples = self._build_examples(5)

    # The BulkInferrer takes in the same Examples used to Trainer the Model,
    # and outputs 5 new examples to be used downstream. This creates additional
    # Examples artifacts in MLMD linked to the Model, but they should NOT be
    # returend as the Examples that the Model was trained on.
    self.put_execution(
        'TFTrainer',
        inputs={
            'examples': self.unwrap_tfx_artifacts(self.examples),
            'model': self.unwrap_tfx_artifacts([self.model]),
        },
        outputs={
            'output_examples': self.unwrap_tfx_artifacts(bulk_inferrer_examples)
        },
    )

    actual = self._training_range([self.model])
    self.assertArtifactListEqual(actual, self.examples[:5])


if __name__ == '__main__':
  tf.test.main()
