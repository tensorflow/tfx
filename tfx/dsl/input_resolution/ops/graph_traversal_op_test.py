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
"""Tests for tfx.dsl.input_resolution.ops.graph_traversal_op."""

from typing import Sequence

import tensorflow as tf
from tfx import types
from tfx.dsl.input_resolution.ops import ops
from tfx.dsl.input_resolution.ops import test_utils


class GraphTraversalOpTest(
    test_utils.ResolverTestCase,
):

  def _graph_traversal(
      self,
      root_artifact: types.Artifact,
      traverse_upstream: bool,
      artifact_type_names=Sequence[str],
  ):
    """Run the GraphTraversal ResolverOp."""
    return self._run_graph_traversal(
        [root_artifact],
        traverse_upstream=traverse_upstream,
        artifact_type_names=artifact_type_names,
    )

  def _run_graph_traversal(self, *args, **kwargs):
    return test_utils.strict_run_resolver_op(
        ops.GraphTraversal,
        args=args,
        kwargs=kwargs,
        store=self.store,
    )

  def setUp(self):
    super().setUp()
    self.init_mlmd()

    spans_and_versions = [(1, 0), (2, 0), (3, 0)]
    self.examples = self.create_examples(spans_and_versions)
    self.transform_graph = self.transform_examples(self.examples)

    # Train using the Examples and TransformGraph.
    self.model = self.prepare_tfx_artifact(test_utils.Model)
    self.train_on_examples(
        model=self.model,
        examples=self.examples,
        transform_graph=self.transform_graph,
    )

    # Bless the Model twice.
    self.model_blessing_1 = self.evaluator_bless_model(self.model)
    self.model_blessing_2 = self.evaluator_bless_model(self.model)

    # Push the Model.
    self.model_push = self.push_model(self.model)

  def testGraphTraversal_NoRootArtifact_ReturnsEmptyDict(self):
    result = self._run_graph_traversal(
        [], traverse_upstream=True, artifact_type_names=['Model']
    )
    self.assertEmpty(result)

  def testGraphTraversal_MultipleRootArtifacts_RaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'does not support batch traversal'):
      self._run_graph_traversal(
          [
              self.prepare_tfx_artifact(test_utils.Model),
              self.prepare_tfx_artifact(test_utils.Model),
          ],
          traverse_upstream=True,
          artifact_type_names=['TransformGraph'],
      )

  def testGraphTraversal_NoArtifactTypeNames_RaisesValueError(self):
    with self.assertRaisesRegex(ValueError, 'artifact_type_names was empty'):
      self._run_graph_traversal(
          [
              self.prepare_tfx_artifact(test_utils.Model),
          ],
          traverse_upstream=True,
          artifact_type_names=[],
      )

  def testGraphTraversal_TraverseUpstream(self):
    # Tests artifacts 2 hops away.
    result = self._graph_traversal(
        self.model,
        traverse_upstream=True,
        artifact_type_names=['TransformGraph', 'Examples'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'Examples': self.examples,
            'TransformGraph': [self.transform_graph],
        },
    )

    # "model_blessing" should have no artifacts in the result dictionary,
    # because there is no path from ModelBlessing --> ModelPush.
    result = self._graph_traversal(
        self.model_push,
        traverse_upstream=True,
        artifact_type_names=[
            'Examples',
            'TransformGraph',
            'Model',
            'ModelBlessing',
        ],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model_push],
            'Examples': self.examples,
            'TransformGraph': [self.transform_graph],
            'Model': [self.model],
            'ModelBlessing': [],
        },
    )

    result = self._graph_traversal(
        self.model,
        traverse_upstream=True,
        artifact_type_names=['TransformGraph', 'FakeArtifactTypeName'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.model],
            'TransformGraph': [self.transform_graph],
            'FakeArtifactTypeName': [],
        },
    )

  def testGraphTraversal_TraverseDownstream(self):
    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Model',
            'TransformGraph',
            'ModelBlessing',
            'ModelPushPath',
        ],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'TransformGraph': [self.transform_graph],
            'Model': [self.model],
            'ModelBlessing': [self.model_blessing_1, self.model_blessing_2],
            'ModelPushPath': [self.model_push],
        },
    )

  def testGraphTraversal_SameArtifactType(self):
    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Examples',
        ],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'Examples': [self.examples[0]],
        },
    )


if __name__ == '__main__':
  tf.test.main()
