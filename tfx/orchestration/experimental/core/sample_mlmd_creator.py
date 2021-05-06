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

from absl import app
from absl import flags

from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import io_utils

from ml_metadata.proto import metadata_store_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('path', '', 'path of mlmd database file')

_PIPELINE_RUN_NUM = 5
_PIPELINE_ID = 'uci-sample-generated'


def _get_mlmd_connection(path: str) -> metadata.Metadata:
  """Returns a MetadataStore for performing MLMD API calls."""
  if os.path.isfile(path):
    raise IOError('File already exists: %s' % path)
  connection_config = metadata.sqlite_metadata_connection_config(path)
  connection_config.sqlite.SetInParent()
  return metadata.Metadata(connection_config=connection_config)


def _test_pipeline(pipeline_id: str, run_id: str):
  """Creates test pipeline with pipeline_id and run_id."""
  pipeline = pipeline_pb2.Pipeline()
  path = os.path.join(
      os.path.dirname(__file__), 'testdata', 'sync_pipeline.pbtxt')
  io_utils.parse_pbtxt_file(path, pipeline)
  pipeline.pipeline_info.id = pipeline_id
  runtime_parameter_utils.substitute_runtime_parameter(pipeline, {
      'pipeline_run_id': run_id,
  })
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


def create_sample_pipeline(m: metadata.Metadata, pipeline_id: str,
                           run_num: int):
  """Creates a list of pipeline and node execution."""
  for i in range(run_num):
    run_id = 'run%02d' % i
    pipeline = _test_pipeline(pipeline_id, run_id)
    pipeline_state = pipeline_ops.initiate_pipeline_start(m, pipeline)
    _execute_nodes(m, pipeline, i)
    if i < run_num - 1:
      execution = pipeline_state.execution
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution])


def main(argv):
  del argv
  with _get_mlmd_connection(FLAGS.path) as m:
    create_sample_pipeline(m, _PIPELINE_ID, _PIPELINE_RUN_NUM)


if __name__ == '__main__':
  app.run(main)
