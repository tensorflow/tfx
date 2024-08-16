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
"""Tests for tfx.dsl.input_resolution.ops.siblings_op."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class SiblingsOpTest(
    test_utils.ResolverTestCase,
):

  def _siblings(
      self,
      root_artifact: types.Artifact,
      output_keys: Sequence[str] = (),
  ):
    """Run the Siblings ResolverOp."""
    return self._run_siblings(
        [root_artifact],
        output_keys=output_keys,
    )

  def _run_siblings(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.Siblings,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    self.spans_and_versions = [(1, 0), (2, 0), (3, 0)]
    self.examples = self.create_examples(self.spans_and_versions)
    self.transform_graph = self.transform_examples(self.examples)

    # Train using the Examples and TransformGraph. The TFTrainer will output a
    # Model and a ModelRunPath.
    self.model = self.prepare_tfx_artifact(test_utils.Model)
    self.model_run = self.prepare_tfx_artifact(test_utils.ModelRun)
    self.put_execution(
        'TFTrainer',
        inputs={
            'examples': self.unwrap_tfx_artifacts(self.examples),
            'transform_graph': self.unwrap_tfx_artifacts(
                [self.transform_graph]
            ),
        },
        outputs={
            'model': self.unwrap_tfx_artifacts([self.model]),
            'model_run': self.unwrap_tfx_artifacts([self.model_run]),
        },
    )

  def testSiblings_NoRootArtifact_ReturnsEmptyDict(self):
    result = self._run_siblings([], output_keys=['model_run'])
    self.assertEmpty(result)

  def testSiblings_MultipleRootArtifacts_RaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'does not support batch queries'):
      self._run_siblings(
          [
              self.prepare_tfx_artifact(test_utils.Model),
              self.prepare_tfx_artifact(test_utils.Model),
          ],
          output_keys=['model_run'],
      )

  def testSiblings_NoOutputKeys(self):
    result = self._siblings(
        self.model,
    )
    self.assertNotIn('model', result)
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'model_run': [self.model_run],
        },
    )

  def testSibling(self):
    result = self._siblings(
        self.model,
        output_keys=['model_run'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'model_run': [self.model_run],
        },
    )

  def testSibling_SameOutputKey(self):
    result = self._siblings(
        self.model,
        output_keys=['model'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'model': [self.model],
        },
    )

  def testSiblingsInvalidOutputKeys(self):
    result = self._siblings(
        self.model,
        output_keys=['examples', 'transform_graph', 'fake_key'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'examples': [],
            'transform_graph': [],
            'fake_key': [],
        },
    )

  def testSiblingsSameOutputArtifactType_DifferentOutputKeys(self):
    data_snapshot = self.create_examples(self.spans_and_versions)
    validation_examples = self.create_examples(self.spans_and_versions)
    output_model = self.prepare_tfx_artifact(test_utils.Model)

    self.put_execution(
        'Component',
        inputs={
            'input_model': self.unwrap_tfx_artifacts([self.model]),
        },
        outputs={
            'output_model': self.unwrap_tfx_artifacts([output_model]),
            'data_snapshot': self.unwrap_tfx_artifacts(data_snapshot),
            'validation_examples': self.unwrap_tfx_artifacts(
                validation_examples
            ),
        },
    )

    # Multiple keys.
    result = self._siblings(
        data_snapshot[0],
        output_keys=[
            'validation_examples',
        ],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [data_snapshot[0]],
            'validation_examples': validation_examples,
        },
    )

    # No output keys.
    result = self._siblings(
        output_model,
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [output_model],
            'data_snapshot': data_snapshot,
            'validation_examples': validation_examples,
        },
    )

  def testSiblings_DescendantArtifactsNotConsideredSiblings(self):
    # Based on:
    #
    # digraph {
    #   artifact_1 -> execution_1
    #   execution_1 -> {artifact_2, root_artifact}
    #   root_artifact -> execution_2 -> artifact_3
    # }

    artifact_1 = self.prepare_tfx_artifact(test_utils.DummyArtifact)
    artifact_2 = self.prepare_tfx_artifact(test_utils.DummyArtifact)
    artifact_3 = self.prepare_tfx_artifact(test_utils.DummyArtifact)
    root_artifact = self.prepare_tfx_artifact(test_utils.DummyArtifact)

    self.put_execution(
        'ComponentA',
        inputs={
            'input_1': self.unwrap_tfx_artifacts([artifact_1]),
        },
        outputs={
            'output_1': self.unwrap_tfx_artifacts([artifact_2]),
            'output_2': self.unwrap_tfx_artifacts([root_artifact]),
        },
    )

    self.put_execution(
        'ComponentB',
        inputs={
            'input_1': self.unwrap_tfx_artifacts([root_artifact]),
        },
        outputs={
            'output_1': self.unwrap_tfx_artifacts([artifact_3]),
        },
    )

    result = self._siblings(
        root_artifact,
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [root_artifact],
            'output_1': [artifact_2],
        },
    )

    result = self._siblings(
        artifact_2,
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [artifact_2],
            'output_2': [root_artifact],
        },
    )
