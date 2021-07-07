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

from typing import Optional
from absl import app
from absl import flags

from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import io_utils

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
  example_gen = test_utils.get_node(pipeline, 'my_example_gen')
  stats_gen = test_utils.get_node(pipeline, 'my_statistics_gen')
  schema_gen = test_utils.get_node(pipeline, 'my_schema_gen')
  transform = test_utils.get_node(pipeline, 'my_transform')
  example_validator = test_utils.get_node(pipeline, 'my_example_validator')
  trainer = test_utils.get_node(pipeline, 'my_trainer')

  test_utils.fake_example_gen_run_with_handle(handle, example_gen, 1, version)
  test_utils.fake_component_output_with_handle(handle, stats_gen, active=False)
  test_utils.fake_component_output_with_handle(handle, schema_gen, active=False)
  test_utils.fake_component_output_with_handle(handle, transform, active=False)
  test_utils.fake_component_output_with_handle(
      handle, example_validator, active=False)
  test_utils.fake_component_output_with_handle(handle, trainer, active=False)


def _get_ir_path(external_ir_file: str):
  if external_ir_file:
    return external_ir_file
  return os.path.join(
      os.path.dirname(__file__), 'testdata', 'sync_pipeline.pbtxt')


def create_sample_pipeline(m: metadata.Metadata,
                           pipeline_id: str,
                           run_num: int,
                           export_ir_path: str = '',
                           external_ir_file: str = '',
                           deployment_config: Optional[message.Message] = None):
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
      _execute_nodes(m, pipeline, i)
    if i < run_num - 1:
      with pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)


def main_factory(mlmd_connection_func):

  def main(argv):
    del argv
    with mlmd_connection_func(FLAGS.path) as m:
      depl_config = pipeline_pb2.IntermediateDeploymentConfig()
      executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
          class_path='fake.ClassPath')
      depl_config.executor_specs['arg1'].Pack(executor_spec)
      depl_config.executor_specs['arg2'].Pack(executor_spec)
      create_sample_pipeline(m, FLAGS.pipeline_id, FLAGS.pipeline_run_num,
                             FLAGS.export_ir_dir, FLAGS.ir_file, depl_config)

  return main


if __name__ == '__main__':
  app.run(main_factory(_get_mlmd_connection))
