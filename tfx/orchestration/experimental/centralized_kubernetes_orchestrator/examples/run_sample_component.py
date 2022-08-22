# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Run sample component (ImportSchemaGen) in Kubernetes, useful for debugging.

Sample command:
```
python tfx/orchestration/experimental/centralized_kubernetes_orchestrator/
examples/run_sample_component.py docker_image={your_docker_image}
job_prefix={your_job_name} container_name={your_container_name}
storage_bucket={your_gcs_bucket_name}
```
"""
from absl import app
from absl import flags
from absl import logging

from tfx import v1 as tfx
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator import kubernetes_job_runner
from tfx.orchestration.portable import data_types
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format

FLAGS = flags.FLAGS
flags.DEFINE_string('docker_image', '', 'docker image')
flags.DEFINE_string('job_prefix', 'sample-job', 'job prefix')
flags.DEFINE_string('container_name', 'centralized-orchestrator',
                    'container name')
flags.DEFINE_string('storage_bucket', '', 'storage bucket')


def _prepare_sample_execution_info(bucket, artifact_path, output_path,
                                   data_path):
  """Prepare sample ImportSchemaGen execution info."""
  pipeline_root = f'gs://{bucket}'
  sample_artifact = tfx.types.standard_artifacts.Schema()
  sample_artifact.uri = pipeline_root + artifact_path

  execution_output_uri = pipeline_root + output_path
  stateful_working_dir = pipeline_root + '/workding/dir'
  exec_properties = {
      'schema_file': pipeline_root + data_path,
  }
  pipeline_info = pipeline_pb2.PipelineInfo(id='my_pipeline')
  pipeline_node = text_format.Parse(
      """
      node_info {
        id: 'my_node'
      }
      """, pipeline_pb2.PipelineNode())

  original = data_types.ExecutionInfo(
      input_dict={},
      output_dict={'schema': [sample_artifact]},
      exec_properties=exec_properties,
      execution_output_uri=execution_output_uri,
      stateful_working_dir=stateful_working_dir,
      pipeline_info=pipeline_info,
      pipeline_node=pipeline_node)

  return original


def _prepare_sample_executable_spec():
  """Prepare sample ImportSchemaGen executable spec."""
  component = tfx.components.ImportSchemaGen.EXECUTOR_SPEC.encode()
  return component


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  execution_info = _prepare_sample_execution_info(FLAGS.storage_bucket,
                                                  '/artifact-output',
                                                  '/test-output',
                                                  '/data/schema.pbtxt')
  executable_spec = _prepare_sample_executable_spec()

  runner = kubernetes_job_runner.KubernetesJobRunner(
      tfx_image=FLAGS.docker_image,
      job_prefix=FLAGS.job_prefix,
      container_name=FLAGS.container_name)
  _ = runner.run(execution_info=execution_info, executable_spec=executable_spec)


if __name__ == '__main__':
  app.run(main)
