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
      artifact_type_names: Sequence[str] = (),
      node_ids: Sequence[str] = (),
      output_keys: Sequence[str] = (),
  ):
    """Run the GraphTraversal ResolverOp."""
    return self._run_graph_traversal(
        [root_artifact],
        traverse_upstream=traverse_upstream,
        artifact_type_names=artifact_type_names,
        node_ids=node_ids,
        output_keys=output_keys,
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

    self.pipeline_name = 'pipeline-name'
    self.pipeline_context = self.put_context('pipeline', self.pipeline_name)

    spans_and_versions = [(1, 0), (2, 0), (3, 0)]
    example_gen_context = self.build_node_context(
        self.pipeline_name, 'example-gen.import-examples'
    )
    self.examples = self.create_examples(
        spans_and_versions,
        contexts=[self.pipeline_context, example_gen_context],
    )

    transform_context = self.build_node_context(
        self.pipeline_name, 'transform.my-transform'
    )
    self.transform_graph = self.transform_examples(
        self.examples,
        contexts=[self.pipeline_context, transform_context],
    )

    # Train using the Examples and TransformGraph.
    trainer_context = self.build_node_context(
        self.pipeline_name, 'tf-trainer.palm-model'
    )
    self.model = self.prepare_tfx_artifact(
        test_utils.Model,
    )
    self.train_on_examples(
        model=self.model,
        examples=self.examples,
        transform_graph=self.transform_graph,
        contexts=[self.pipeline_context, trainer_context],
    )

    # Bless the Model twice.
    evaluator_context = self.build_node_context(
        self.pipeline_name, 'evaluator.my-evaluator'
    )
    self.model_blessing_1 = self.evaluator_bless_model(
        self.model,
        contexts=[self.pipeline_context, evaluator_context],
    )
    self.model_blessing_2 = self.evaluator_bless_model(
        self.model,
        contexts=[self.pipeline_context, evaluator_context],
    )

    # Push the Model.
    pusher_context = self.build_node_context(
        self.pipeline_name, 'servomatic-pusher.my-pusher'
    )
    self.model_push = self.push_model(
        self.model,
        contexts=[self.pipeline_context, pusher_context],
    )

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

  def testGraphTraversal_NodeIds_OutputKeys(self):
    model_2 = self.prepare_tfx_artifact(
        test_utils.Model,
    )
    model_3 = self.prepare_tfx_artifact(
        test_utils.Model,
    )

    custom_trainer_context = self.build_node_context(
        self.pipeline_name, 'tf-trainer.student-model'
    )
    self.put_execution(
        'CustomTrainer',
        inputs={
            'examples': self.unwrap_tfx_artifacts(self.examples),
            'transform_graph': self.unwrap_tfx_artifacts(
                [self.transform_graph]
            ),
        },
        outputs={
            'parent_model': self.unwrap_tfx_artifacts([model_2]),
            'child_model': self.unwrap_tfx_artifacts([model_3]),
        },
        contexts=[self.pipeline_context, custom_trainer_context],
    )

    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Model',
        ],
        node_ids=['tf-trainer.palm-model'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'Model': [self.model],
        },
    )

    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Model',
        ],
        node_ids=['tf-trainer.palm-model'],
        output_keys=['parent_model'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'Model': [],
        },
    )

    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Model',
        ],
        node_ids=['tf-trainer.student-model'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'Model': [model_2, model_3],
        },
    )

    result = self._graph_traversal(
        self.examples[0],
        traverse_upstream=False,
        artifact_type_names=[
            'Model',
        ],
        node_ids=['tf-trainer.student-model'],
        output_keys=['child_model'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [self.examples[0]],
            'Model': [model_3],
        },
    )

    result = self._graph_traversal(
        model_3,
        traverse_upstream=True,
        artifact_type_names=[
            'TransformGraph',
        ],
        node_ids=['transform.my-transform'],
        output_keys=['transform_graph'],
    )
    self.assertArtifactMapsEqual(
        result,
        {
            'root_artifact': [model_3],
            'TransformGraph': [self.transform_graph],
        },
    )
