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
"""Creates testing MLMD with TFX data model."""
import os
import tempfile

from typing import Optional, Callable
from absl import app
from absl import flags

from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_sync_pipeline
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import io_utils
from tfx.utils import status as status_lib

from google.protobuf import message
from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('ir_file', '', 'path of ir file to create sample mlmd')
flags.DEFINE_string('path', '', 'path of mlmd database file')
flags.DEFINE_string('export_ir_dir', '', 'directory path of output IR files')
flags.DEFINE_integer('pipeline_run_num', 5, 'number of pipeline run')
flags.DEFINE_string('pipeline_id', 'uci-sample-generated', 'id of pipeline')


def _get_mlmd_connection(path: str) -> metadata.Metadata:
  """Returns a MetadataStore for performing MLMD API calls."""
  if os.path.isfile(path):
    raise IOError('File already exists: %s' % path)
  connection_config = metadata.sqlite_metadata_connection_config(path)
  connection_config.sqlite.SetInParent()
  return metadata.Metadata(connection_config=connection_config)


def _test_pipeline(ir_path: str, pipeline_id: str, run_id: str,
                   deployment_config: Optional[message.Message]):
  """Creates test pipeline with pipeline_id and run_id."""
  pipeline = pipeline_pb2.Pipeline()
  io_utils.parse_pbtxt_file(ir_path, pipeline)
  pipeline.pipeline_info.id = pipeline_id
  runtime_parameter_utils.substitute_runtime_parameter(pipeline, {
      constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id,
  })
  if deployment_config:
    pipeline.deployment_config.Pack(deployment_config)
  return pipeline


def _execute_nodes(handle: metadata.Metadata, pipeline: pipeline_pb2.Pipeline,
                   version: int):
  """Creates fake execution of nodes."""
  for node in pstate.get_all_nodes(pipeline):
    if node.node_info.id == 'my_example_gen':
      test_utils.fake_example_gen_run_with_handle(handle, node, 1, version)
    else:
      test_utils.fake_component_output_with_handle(handle, node, active=False)
    pipeline_state = test_utils.get_or_create_pipeline_state(handle, pipeline)
    with pipeline_state:
      with pipeline_state.node_state_update_context(
          task_lib.NodeUid.from_node(pipeline, node)
      ) as node_state:
        node_state.update(
            pstate.NodeState.COMPLETE,
            status_lib.Status(code=status_lib.Code.OK, message='all ok'),
        )


def _get_ir_path(external_ir_file: str):
  if external_ir_file:
    return external_ir_file
  ir_file_path = tempfile.mktemp(suffix='.pbtxt')
  io_utils.write_pbtxt_file(ir_file_path, test_sync_pipeline.create_pipeline())
  return ir_file_path


def create_sample_pipeline(m: metadata.Metadata,
                           pipeline_id: str,
                           run_num: int,
                           export_ir_path: str = '',
                           external_ir_file: str = '',
                           deployment_config: Optional[message.Message] = None,
                           execute_nodes_func: Callable[
                               [metadata.Metadata, pipeline_pb2.Pipeline, int],
                               None] = _execute_nodes):
  """Creates a list of pipeline and node execution."""
  ir_path = _get_ir_path(external_ir_file)
  for i in range(run_num):
    run_id = 'run%02d' % i
    pipeline = _test_pipeline(ir_path, pipeline_id, run_id, deployment_config)
    if export_ir_path:
      output_path = os.path.join(export_ir_path,
                                 '%s_%s.pbtxt' % (pipeline_id, run_id))
      io_utils.write_pbtxt_file(output_path, pipeline)
    pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
    if not external_ir_file:
      execute_nodes_func(m, pipeline, i)
    if i < run_num - 1:
      with pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)


def main_factory(mlmd_connection_func: Callable[[str], metadata.Metadata],
                 execute_nodes_func: Callable[
                     [metadata.Metadata, pipeline_pb2.Pipeline, int],
                     None] = _execute_nodes):

  def main(argv):
    del argv
    with mlmd_connection_func(FLAGS.path) as m:
      depl_config = pipeline_pb2.IntermediateDeploymentConfig()
      executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
          class_path='fake.ClassPath')
      depl_config.executor_specs['arg1'].Pack(executor_spec)
      depl_config.executor_specs['arg2'].Pack(executor_spec)
      create_sample_pipeline(m, FLAGS.pipeline_id, FLAGS.pipeline_run_num,
                             FLAGS.export_ir_dir, FLAGS.ir_file, depl_config,
                             execute_nodes_func)

  return main


if __name__ == '__main__':
  app.run(main_factory(_get_mlmd_connection))
