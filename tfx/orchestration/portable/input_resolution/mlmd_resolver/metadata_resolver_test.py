# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for metadata resolver."""
from typing import Dict, List
from absl.testing import absltest
from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver
from tfx.orchestration.portable.input_resolution.mlmd_resolver import metadata_resolver_utils
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2


def create_artifact_type(
    store: mlmd.MetadataStore, typename: str
) -> metadata_store_pb2.ArtifactType:
  """Put an Artifact Type in the MLMD database."""
  artifact_type = metadata_store_pb2.ArtifactType(name=typename)
  artifact_type.id = store.put_artifact_type(artifact_type)
  return artifact_type


def create_artifact(
    store: mlmd.MetadataStore, artifact_type_id: int, name: str
) -> metadata_store_pb2.Artifact:
  """Put an Artifact in the MLMD database."""
  artifact = metadata_store_pb2.Artifact(
      name=name, type_id=artifact_type_id, uri=f'https://{name}'
  )
  [artifact.id] = store.put_artifacts([artifact])

  return artifact


def create_execution_type(
    store: mlmd.MetadataStore, typename: str
) -> metadata_store_pb2.ExecutionType:
  """Put an Execution Type in the MLMD database."""
  execution_type = metadata_store_pb2.ExecutionType(name=typename)
  execution_type.id = store.put_execution_type(execution_type)
  return execution_type


def create_execution(
    store: mlmd.MetadataStore,
    execution_type_id: int,
    name: str,
    inputs: Dict[str, List[metadata_store_pb2.Artifact]],
    outputs: Dict[str, List[metadata_store_pb2.Artifact]],
    contexts: List[metadata_store_pb2.Context],
    output_event_type: metadata_store_pb2.Event.Type = metadata_store_pb2.Event.OUTPUT,
) -> metadata_store_pb2.Execution:
  """Put an Execution in the MLMD database.

  Args:
      store: metadata store
      execution_type_id: type id of the execution
      name: name of the execution
      inputs: a mapping of the event step key to a list of input artifacts.
      outputs: a mapping of the event step key to a list of output artifacts.
      contexts: a list of contexts that the execution is associated with.
      output_event_type: the event type of all output events. It must be one of
        the valid output event types.

  Returns:
  Created execution.
  """
  if output_event_type not in metadata_resolver_utils.OUTPUT_EVENT_TYPES:
    raise ValueError(f'{output_event_type} is not a valid output event type.')
  execution = metadata_store_pb2.Execution(
      type_id=execution_type_id,
      name=name,
  )
  artifact_and_events = []
  for input_key, artifacts in inputs.items():
    for i, artifact in enumerate(artifacts):
      event = metadata_store_pb2.Event(
          type=metadata_store_pb2.Event.INPUT, artifact_id=artifact.id
      )
      event.path.steps.add().key = input_key
      event.path.steps.add().index = i
      artifact_and_events.append((artifact, event))
  for output_key, artifacts in outputs.items():
    for i, artifact in enumerate(artifacts):
      event = metadata_store_pb2.Event(
          type=output_event_type, artifact_id=artifact.id
      )
      event.path.steps.add().key = output_key
      event.path.steps.add().index = i
      artifact_and_events.append((artifact, event))
  execution.id, _, _ = store.put_execution(
      execution, artifact_and_events, contexts
  )
  return execution


def create_context_type(
    store: mlmd.MetadataStore, typename: str
) -> metadata_store_pb2.ContextType:
  """Put a Context Type in the MLMD database."""
  context_type = metadata_store_pb2.ContextType(name=typename)
  context_type.id = store.put_context_type(context_type)
  return context_type


def create_context(
    store: mlmd.MetadataStore, context_type_id: int, context_name: str
) -> metadata_store_pb2.Context:
  """Put a Context in the MLMD database."""

  context = metadata_store_pb2.Context(
      type_id=context_type_id, name=context_name
  )
  [context.id] = store.put_contexts([context])
  return context


class MetadataResolverTest(absltest.TestCase):

  def setUp(self):
    """Create and insert a lineage graph in metadata store.

    ExampleGen-1     ExampleGen-2     ExampleGen-3
         │                │                │
         ▼                ▼                ▼
     Example-1       Example-2       Example-3
         │              │ │              │ │
         └─────┬────────┘ └─────┬────────┘ │
               ▼                ▼          │
           Trainer-1        Trainer-2      │
               │                │          │
               ▼                ▼          │
             Model-1          Model-2      │
               │                           │
               └───────────────────────┐   │
                                       ▼   ▼
                                   Evaluator-1
                                       │
                                       ▼
                                   Evaluation-1
    """
    super().setUp()
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.fake_database.SetInParent()
    self.store = mlmd.MetadataStore(connection_config)

    self._mlmd_connection_manager = None

    self.resolver = metadata_resolver.MetadataResolver(
        self.store, mlmd_connection_manager=self._mlmd_connection_manager
    )

    self.exp_type = create_artifact_type(self.store, 'Examples')
    self.example_gen_type = create_execution_type(self.store, 'ExampleGen')
    self.trainer_type = create_execution_type(self.store, 'Trainer')
    self.model_type = create_artifact_type(self.store, 'Model')
    self.evaluator_type = create_execution_type(self.store, 'Evaluator')
    self.evaluation_type = create_artifact_type(self.store, 'Evaluation')
    self.pipe_type = create_context_type(self.store, 'pipeline')
    self.run_type = create_context_type(self.store, 'pipeline_run')
    self.node_type = create_context_type(self.store, 'node')

    self.pipe_ctx = create_context(self.store, self.pipe_type.id, 'my-pipeline')
    self.run1_ctx = create_context(
        self.store, self.run_type.id, 'my-pipeline.run-01'
    )
    self.run2_ctx = create_context(
        self.store, self.run_type.id, 'my-pipeline.run-02'
    )
    self.run3_ctx = create_context(
        self.store, self.run_type.id, 'my-pipeline.run-03'
    )
    self.example_gen_ctx = create_context(
        self.store, self.node_type.id, 'my-pipeline.ExampleGen'
    )
    self.trainer_ctx = create_context(
        self.store, self.node_type.id, 'my-pipeline.Trainer'
    )
    self.evaluator_ctx = create_context(
        self.store, self.node_type.id, 'my-pipeline.Evaluator'
    )
    self.e1 = create_artifact(self.store, self.exp_type.id, name='Example-1')
    self.e2 = create_artifact(self.store, self.exp_type.id, name='Example-2')
    self.e3 = create_artifact(self.store, self.exp_type.id, name='Example-3')
    self.m1 = create_artifact(self.store, self.model_type.id, name='Model-1')
    self.m2 = create_artifact(self.store, self.model_type.id, name='Model-2')
    self.ev1 = create_artifact(
        self.store, self.evaluation_type.id, name='Evaluation-1'
    )

    self.expgen1 = create_execution(
        self.store,
        self.example_gen_type.id,
        name='ExampleGen-1',
        inputs={},
        outputs={'examples': [self.e1]},
        contexts=[self.pipe_ctx, self.run1_ctx, self.example_gen_ctx],
    )
    self.expgen2 = create_execution(
        self.store,
        self.example_gen_type.id,
        name='ExampleGen-2',
        inputs={},
        outputs={'examples': [self.e2]},
        contexts=[self.pipe_ctx, self.run2_ctx, self.example_gen_ctx],
    )
    self.expgen3 = create_execution(
        self.store,
        self.example_gen_type.id,
        name='ExampleGen-3',
        inputs={},
        outputs={'examples': [self.e3]},
        contexts=[self.pipe_ctx, self.run3_ctx, self.example_gen_ctx],
    )
    self.trainer1 = create_execution(
        self.store,
        self.trainer_type.id,
        name='Trainer-1',
        inputs={'examples': [self.e1, self.e2]},
        outputs={'model': [self.m1]},
        contexts=[self.pipe_ctx, self.run1_ctx, self.trainer_ctx],
    )
    self.trainer2 = create_execution(
        self.store,
        self.trainer_type.id,
        name='Trainer-2',
        inputs={'examples': [self.e2, self.e3]},
        outputs={'model': [self.m2]},
        contexts=[self.pipe_ctx, self.run2_ctx, self.trainer_ctx],
        output_event_type=metadata_store_pb2.Event.Type.PENDING_OUTPUT,
    )
    self.evaluator = create_execution(
        self.store,
        self.evaluator_type.id,
        name='Evaluator-1',
        inputs={'examples': [self.e3], 'model': [self.m1]},
        outputs={'evaluation': [self.ev1]},
        contexts=[self.pipe_ctx, self.run3_ctx, self.evaluator_ctx],
    )



  def test_get_downstream_artifacts_by_artifact_ids(self):
    # Test: get downstream artifacts by example_1, with max_num_hops = 0
    result_from_exp1 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id], max_num_hops=0
    )
    self.assertLen(result_from_exp1, 1)
    self.assertIn(self.e1.id, result_from_exp1)
    self.assertCountEqual(
        [result_from_exp1[self.e1.id][0][0].name], [self.e1.name]
    )

    # Test: get downstream artifacts by example_1, with max_num_hops = 2
    result_from_exp1 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id], max_num_hops=2
    )
    self.assertLen(result_from_exp1, 1)
    self.assertIn(self.e1.id, result_from_exp1)
    self.assertCountEqual(
        [(e.name, t.name) for e, t in result_from_exp1[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
        ],
    )

    # Test: get downstream artifacts by example_1, with max_num_hops = 20
    result_from_exp1 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id], max_num_hops=20
    )
    self.assertLen(result_from_exp1, 1)
    self.assertIn(self.e1.id, result_from_exp1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )

    # Test: get downstream artifacts by example_1, with max_num_hops
    # unspecified.
    result_from_exp1 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id], max_num_hops=20
    )
    self.assertLen(result_from_exp1, 1)
    self.assertIn(self.e1.id, result_from_exp1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )

    # Test: get downstream artifacts by [example_1, example_2, example_3],
    # with max_num_hops = 20
    result_from_exp123 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id], max_num_hops=20
    )
    self.assertCountEqual(
        [self.e1.id, self.e2.id, self.e3.id], result_from_exp123
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp123[self.e2.id]],
        [
            (self.e2.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.m2.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp123[self.e3.id]],
        [
            (self.e3.name, self.exp_type.name),
            (self.m2.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    # Test: get empty result if `artifact_ids` is empty.
    self.assertEmpty(self.resolver.get_downstream_artifacts_by_artifact_ids([]))

  def test_get_downstream_artifacts_by_artifact_uri(self):
    # Test: get downstream artifacts by example_2, with max_num_hops = 0
    result_from_exp2 = self.resolver.get_downstream_artifacts_by_artifact_uri(
        self.e2.uri, max_num_hops=0
    )
    self.assertLen(result_from_exp2, 1)
    self.assertIn(self.e2.id, result_from_exp2)
    self.assertCountEqual(
        [result_from_exp2[self.e2.id][0].name], [self.e2.name]
    )

    # Test: get downstream artifacts by example_2, with max_num_hops = 2
    result_from_exp2 = self.resolver.get_downstream_artifacts_by_artifact_uri(
        self.e2.uri, max_num_hops=2
    )
    self.assertLen(result_from_exp2, 1)
    self.assertIn(self.e2.id, result_from_exp2)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_exp2[self.e2.id]],
        [self.e2.name, self.m1.name, self.m2.name],
    )

    # Test: get downstream artifacts by example_2, with max_num_hops = 20
    result_from_exp2 = self.resolver.get_downstream_artifacts_by_artifact_uri(
        self.e2.uri, max_num_hops=20
    )
    self.assertLen(result_from_exp2, 1)
    self.assertIn(self.e2.id, result_from_exp2)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_exp2[self.e2.id]],
        [self.e2.name, self.m1.name, self.m2.name, self.ev1.name],
    )

    # Test: get downstream artifacts by example_2, with max_num_hops
    # unspecified.
    result_from_exp2 = self.resolver.get_downstream_artifacts_by_artifact_uri(
        self.e2.uri
    )
    self.assertLen(result_from_exp2, 1)
    self.assertIn(self.e2.id, result_from_exp2)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_exp2[self.e2.id]],
        [self.e2.name, self.m1.name, self.m2.name, self.ev1.name],
    )

    # Test: raise ValueError if `artifact_uri` is empty.
    with self.assertRaisesRegex(ValueError, '`artifact_uri` is empty.'):
      self.resolver.get_downstream_artifacts_by_artifact_uri('')

  def test_get_filtered_downstream_artifacts_by_artifact_ids(self):
    # Test: get downstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact name.
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name = "{self.e1.name}" ',
    )
    self.assertLen(result_from_exps, 1)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertCountEqual(
        [result_from_exps[self.e1.id][0][0].name], [self.e1.name]
    )

    # Test: get downstream artifacts by examples, with max_num_hops = 1, filter
    # by artifact name.
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=1,
        filter_query=f'name = "{self.e1.name}" ',
    )
    self.assertLen(result_from_exps, 1)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertCountEqual(
        [result_from_exps[self.e1.id][0][0].name], [self.e1.name]
    )

    # Test: get downstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact type = Example.
    artifact_names_filter_query = '","'.join(
        [self.e1.name, self.e2.name, self.e3.name]
    )
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name IN ("{artifact_names_filter_query}")',
    )
    self.assertLen(result_from_exps, 3)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertIn(self.e2.id, result_from_exps)
    self.assertIn(self.e3.id, result_from_exps)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e1.id]],
        [(self.e1.name, self.exp_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e2.id]],
        [(self.e2.name, self.exp_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e3.id]],
        [(self.e3.name, self.exp_type.name)],
    )

    # Test: get downstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact type = Evaluation.
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name = "{self.evaluation_type.name}"',
    )
    self.assertEmpty(result_from_exps)

    # Test: get downstream artifacts by examples, with max_num_hops = 20, filter
    # by artifact type.
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=20,
        filter_query=f'type = "{self.model_type.name}"',
    )
    self.assertLen(result_from_exps, 3)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertIn(self.e2.id, result_from_exps)
    self.assertIn(self.e3.id, result_from_exps)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e1.id]],
        [(self.m1.name, self.model_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e2.id]],
        [
            (self.m1.name, self.model_type.name),
            (self.m2.name, self.model_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e3.id]],
        [(self.m2.name, self.model_type.name)],
    )

    # Test: get downstream artifacts by examples and evaluation, with
    # max_num_hops = 20, filter by artifact type = Model or Evaluation.
    result_from_exps_eva = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id, self.ev1.id],
        max_num_hops=20,
        filter_query=(
            f'type = "{self.model_type.name}" OR type ='
            f' "{self.evaluation_type.name}"'
        ),
    )
    self.assertLen(result_from_exps_eva, 4)
    self.assertIn(self.e1.id, result_from_exps_eva)
    self.assertIn(self.e2.id, result_from_exps_eva)
    self.assertIn(self.e3.id, result_from_exps_eva)
    self.assertIn(self.ev1.id, result_from_exps_eva)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e1.id]],
        [
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e2.id]],
        [
            (self.m1.name, self.model_type.name),
            (self.m2.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e3.id]],
        [
            (self.m2.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.ev1.id]],
        [(self.ev1.name, self.evaluation_type.name)],
    )

    # Test: get downstream artifacts by examples and evaluation, with
    # max_num_hops = 20, filter by artifact type = Model.
    result_from_exps_eva = (
        self.resolver.get_downstream_artifacts_by_artifact_ids(
            [self.e1.id, self.e2.id, self.e3.id],
            max_num_hops=20,
            filter_query=f'type = "{self.model_type.name}"',
        )
    )
    self.assertLen(result_from_exps_eva, 3)
    self.assertIn(self.e1.id, result_from_exps_eva)
    self.assertIn(self.e2.id, result_from_exps_eva)
    self.assertIn(self.e3.id, result_from_exps_eva)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e1.id]],
        [(self.m1.name, self.model_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e2.id]],
        [
            (self.m1.name, self.model_type.name),
            (self.m2.name, self.model_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_eva[self.e3.id]],
        [(self.m2.name, self.model_type.name)],
    )

    # Test: get downstream artifacts by example_1, with max_num_hops and
    # filter_query unspecified.
    result_from_exp1 = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id]
    )
    self.assertLen(result_from_exp1, 1)
    self.assertIn(self.e1.id, result_from_exp1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )

    # Test: get downstream artifacts by examples, filter events by event type.
    # model_2 will be excluded from downstream artifacts list for example_2 and
    # example_3.
    def _is_input_event_or_valid_output_event(
        event: metadata_store_pb2.Event,
    ) -> bool:
      return event.type != metadata_store_pb2.Event.Type.PENDING_OUTPUT

    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=20,
        event_filter=_is_input_event_or_valid_output_event,
    )
    self.assertLen(result_from_exps, 3)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertIn(self.e2.id, result_from_exps)
    self.assertIn(self.e3.id, result_from_exps)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e2.id]],
        [
            (self.e2.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e3.id]],
        [
            (self.e3.name, self.exp_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )

    # Test: get downstream artifacts by examples, filter events by event type
    # and filter the downstream artifacts by artifact_type = Model.
    # model_2 will be excluded from downstream artifacts list for example_2 and
    # example_3. As example_3 has no qualified downstream artifacts, it's not
    # included in the result.
    result_from_exps = self.resolver.get_downstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=20,
        filter_query=f'type = "{self.model_type.name}"',
        event_filter=_is_input_event_or_valid_output_event,
    )
    self.assertLen(result_from_exps, 2)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertIn(self.e2.id, result_from_exps)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e1.id]],
        [(self.m1.name, self.model_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e2.id]],
        [(self.m1.name, self.model_type.name)],
    )


  def test_get_upstream_artifacts_by_artifact_ids(self):
    # Test: get upstream artifacts by model_1, with max_num_hops = 0
    result_from_m1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.m1.id], max_num_hops=0
    )
    self.assertLen(result_from_m1, 1)
    self.assertIn(self.m1.id, result_from_m1)
    self.assertCountEqual(
        [result_from_m1[self.m1.id][0][0].name], [self.m1.name]
    )

    # Test: get upstream artifacts by model_1, with max_num_hops = 2
    result_from_m1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.m1.id], max_num_hops=2
    )
    self.assertLen(result_from_m1, 1)
    self.assertIn(self.m1.id, result_from_m1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_m1[self.m1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.e2.name, self.exp_type.name),
        ],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops = 2
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.ev1.id], max_num_hops=2
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_ev1[self.ev1.id]],
        [
            (self.ev1.name, self.evaluation_type.name),
            (self.e3.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
        ],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops = 20
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.ev1.id], max_num_hops=20
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_ev1[self.ev1.id]],
        [
            (self.ev1.name, self.evaluation_type.name),
            (self.e3.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.e1.name, self.exp_type.name),
            (self.e2.name, self.exp_type.name),
        ],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops
    # unspecified.
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.ev1.id]
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_ev1[self.ev1.id]],
        [
            (self.ev1.name, self.evaluation_type.name),
            (self.e3.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.e1.name, self.exp_type.name),
            (self.e2.name, self.exp_type.name),
        ],
    )

    # Test: get upstream artifacts by example_1, evaluation_1, with max_num_hops
    # = 20.
    result_from_exp1_ev1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.ev1.id], max_num_hops=20
    )
    self.assertLen(result_from_exp1_ev1, 2)
    self.assertIn(self.e1.id, result_from_exp1_ev1)
    self.assertIn(self.ev1.id, result_from_exp1_ev1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1_ev1[self.e1.id]],
        [(self.e1.name, self.exp_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exp1_ev1[self.ev1.id]],
        [
            (self.ev1.name, self.evaluation_type.name),
            (self.e3.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.e1.name, self.exp_type.name),
            (self.e2.name, self.exp_type.name),
        ],
    )
    # Test: get empty result if `artifact_ids` is empty.
    self.assertEmpty(self.resolver.get_upstream_artifacts_by_artifact_ids([]))

  def test_get_upstream_artifacts_by_artifact_uri(self):
    # Test: get upstream artifacts by model_1, with max_num_hops = 0
    result_from_m1 = self.resolver.get_upstream_artifacts_by_artifact_uri(
        self.m1.uri, max_num_hops=0
    )
    self.assertLen(result_from_m1, 1)
    self.assertIn(self.m1.id, result_from_m1)
    self.assertEqual([result_from_m1[self.m1.id][0].name], [self.m1.name])

    # Test: get upstream artifacts by model_1, with max_num_hops = 2
    result_from_m1 = self.resolver.get_upstream_artifacts_by_artifact_uri(
        self.m1.uri, max_num_hops=2
    )
    self.assertLen(result_from_m1, 1)
    self.assertIn(self.m1.id, result_from_m1)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_m1[self.m1.id]],
        [self.e1.name, self.m1.name, self.e2.name],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops = 2
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_uri(
        self.ev1.uri, max_num_hops=2
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_ev1[self.ev1.id]],
        [self.ev1.name, self.e3.name, self.m1.name],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops = 20
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_uri(
        self.ev1.uri, max_num_hops=20
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_ev1[self.ev1.id]],
        [self.ev1.name, self.e3.name, self.m1.name, self.e1.name, self.e2.name],
    )

    # Test: get upstream artifacts by evaluation_1, with max_num_hops
    # unspecified.
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_uri(
        self.ev1.uri
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [artifact.name for artifact in result_from_ev1[self.ev1.id]],
        [self.ev1.name, self.e3.name, self.m1.name, self.e1.name, self.e2.name],
    )
    # Test: raise ValueError if `artifact_uri` is empty.
    with self.assertRaisesRegex(ValueError, '`artifact_uri` is empty.'):
      self.resolver.get_upstream_artifacts_by_artifact_uri('')

  def test_get_filtered_upstream_artifacts_by_artifact_ids(self):
    # Test: get upstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact name.
    result_from_exps = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name = "{self.e1.name}" ',
    )
    self.assertLen(result_from_exps, 1)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertCountEqual(
        [result_from_exps[self.e1.id][0][0].name], [self.e1.name]
    )

    # Test: get upstream artifacts by examples, with max_num_hops = 1, filter
    # by artifact name.
    result_from_exps = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=1,
        filter_query=f'name = "{self.e1.name}" ',
    )
    self.assertLen(result_from_exps, 1)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertCountEqual(
        [result_from_exps[self.e1.id][0][0].name], [self.e1.name]
    )

    # Test: get upstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact type = Example.
    artifact_names_filter_query = '","'.join(
        [self.e1.name, self.e2.name, self.e3.name]
    )
    result_from_exps = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name IN ("{artifact_names_filter_query}")',
    )
    self.assertLen(result_from_exps, 3)
    self.assertIn(self.e1.id, result_from_exps)
    self.assertIn(self.e2.id, result_from_exps)
    self.assertIn(self.e3.id, result_from_exps)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e1.id]],
        [(self.e1.name, self.exp_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e2.id]],
        [(self.e2.name, self.exp_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps[self.e3.id]],
        [(self.e3.name, self.exp_type.name)],
    )

    # Test: get upstream artifacts by examples, with max_num_hops = 0, filter
    # by artifact type = Evaluation.
    result_from_exps = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.e2.id, self.e3.id],
        max_num_hops=0,
        filter_query=f'name = "{self.evaluation_type.name}"',
    )
    self.assertEmpty(result_from_exps)

    # Test: get upstream artifacts by evaluation, with max_num_hops = 20, filter
    # by artifact type.
    result_from_eva = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.ev1.id],
        max_num_hops=20,
        filter_query=f'type = "{self.model_type.name}"',
    )
    self.assertLen(result_from_eva, 1)
    self.assertIn(self.ev1.id, result_from_eva)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_eva[self.ev1.id]],
        [(self.m1.name, self.model_type.name)],
    )

    # Test: get upstream artifacts by examples, models and evaluation, with
    # max_num_hops = 20, filter by artifact type = Model or Evaluation.
    result_from_exps_model_eva = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.e1.id, self.m2.id, self.ev1.id],
        max_num_hops=20,
        filter_query=(
            f'type = "{self.model_type.name}" OR type ='
            f' "{self.evaluation_type.name}"'
        ),
    )
    self.assertLen(result_from_exps_model_eva, 2)
    self.assertIn(self.m2.id, result_from_exps_model_eva)
    self.assertIn(self.ev1.id, result_from_exps_model_eva)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_model_eva[self.m2.id]],
        [(self.m2.name, self.model_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_exps_model_eva[self.ev1.id]],
        [
            (self.ev1.name, self.evaluation_type.name),
            (self.m1.name, self.model_type.name),
        ],
    )

    # Test: get upstream artifacts by evaluation, with max_num_hops and
    # filter_query unspecified.
    result_from_ev1 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.ev1.id]
    )
    self.assertLen(result_from_ev1, 1)
    self.assertIn(self.ev1.id, result_from_ev1)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_ev1[self.ev1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.e2.name, self.exp_type.name),
            (self.e3.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
            (self.ev1.name, self.evaluation_type.name),
        ],
    )

    def _is_input_event_or_valid_output_event(
        event: metadata_store_pb2.Event,
    ) -> bool:
      return event.type != metadata_store_pb2.Event.Type.PENDING_OUTPUT

    # Test: get upstream artifacts filtered by events from models. Only
    # artifacts connected to model_1 and model_2 itself will be included.
    result_from_m12 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.m1.id, self.m2.id],
        max_num_hops=20,
        event_filter=_is_input_event_or_valid_output_event,
    )
    self.assertLen(result_from_m12, 2)
    self.assertIn(self.m1.id, result_from_m12)
    self.assertIn(self.m2.id, result_from_m12)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_m12[self.m1.id]],
        [
            (self.e1.name, self.exp_type.name),
            (self.e2.name, self.exp_type.name),
            (self.m1.name, self.model_type.name),
        ],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_m12[self.m2.id]],
        [(self.m2.name, self.model_type.name)],
    )

    # Test: get upstream artifacts filtered by events from models, with filter
    # query for filtering upstream artifacts with type = Model. Only model_1
    # and model_2 will included.
    result_from_m12 = self.resolver.get_upstream_artifacts_by_artifact_ids(
        [self.m1.id, self.m2.id],
        max_num_hops=20,
        filter_query=f'type = "{self.model_type.name}"',
        event_filter=_is_input_event_or_valid_output_event,
    )
    self.assertLen(result_from_m12, 2)
    self.assertIn(self.m1.id, result_from_m12)
    self.assertIn(self.m2.id, result_from_m12)
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_m12[self.m1.id]],
        [(self.m1.name, self.model_type.name)],
    )
    self.assertCountEqual(
        [(a.name, t.name) for a, t in result_from_m12[self.m2.id]],
        [(self.m2.name, self.model_type.name)],
    )
