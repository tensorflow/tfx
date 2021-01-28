# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Test utilities."""

from absl.testing.absltest import mock
from tfx import types
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib


def fake_example_gen_run(mlmd_connection, example_gen, span, version):
  """Writes fake example_gen output and successful execution to MLMD."""
  with mlmd_connection as m:
    output_example = types.Artifact(
        example_gen.outputs.outputs['output_examples'].artifact_spec.type)
    output_example.set_int_custom_property('span', span)
    output_example.set_int_custom_property('version', version)
    output_example.uri = 'my_examples_uri'
    contexts = context_lib.prepare_contexts(m, example_gen.contexts)
    execution = execution_publish_utils.register_execution(
        m, example_gen.node_info.type, contexts)
    execution_publish_utils.publish_succeeded_execution(
        m, execution.id, contexts, {
            'output_examples': [output_example],
        })


def fake_transform_output(mlmd_connection, transform, execution=None):
  """Writes fake transform output and execution to MLMD."""
  with mlmd_connection as m:
    output_transform_graph = types.Artifact(
        transform.outputs.outputs['transform_graph'].artifact_spec.type)
    output_transform_graph.uri = 'my_transform_graph_uri'
    contexts = context_lib.prepare_contexts(m, transform.contexts)
    if not execution:
      execution = execution_publish_utils.register_execution(
          m, transform.node_info.type, contexts)
    execution_publish_utils.publish_succeeded_execution(
        m, execution.id, contexts, {
            'transform_graph': [output_transform_graph],
        })


def fake_trainer_output(mlmd_connection, trainer, execution=None, active=False):
  """Writes fake trainer output and execution to MLMD."""
  with mlmd_connection as m:
    output_trainer_model = types.Artifact(
        trainer.outputs.outputs['model'].artifact_spec.type)
    output_trainer_model.uri = 'my_trainer_model_uri'
    contexts = context_lib.prepare_contexts(m, trainer.contexts)
    if not execution:
      execution = execution_publish_utils.register_execution(
          m, trainer.node_info.type, contexts)
    if not active:
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'model': [output_trainer_model],
          })


def create_exec_node_task(node_uid,
                          execution=None,
                          contexts=None,
                          exec_properties=None,
                          input_artifacts=None,
                          output_artifacts=None,
                          executor_output_uri=None,
                          stateful_working_dir=None,
                          pipeline=None,
                          is_cancelled=False) -> task_lib.ExecNodeTask:
  """Creates an `ExecNodeTask` for testing."""
  return task_lib.ExecNodeTask(
      node_uid=node_uid,
      execution=execution or mock.Mock(),
      contexts=contexts or [],
      exec_properties=exec_properties or {},
      input_artifacts=input_artifacts or {},
      output_artifacts=output_artifacts or {},
      executor_output_uri=executor_output_uri or '',
      stateful_working_dir=stateful_working_dir or '',
      pipeline=pipeline or mock.Mock(),
      is_cancelled=is_cancelled)
