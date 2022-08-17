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
"""Client for orchestrator.

A simple client to communicate with the orchestrator server.
"""

import datetime

from absl import app
from absl import flags
import grpc
from tfx import v1 as tfx
from tfx.dsl.compiler import compiler
from tfx.dsl.compiler import constants
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc
from tfx.orchestration.portable import runtime_parameter_utils

# Flags to use in the command line to specifiy the port and the msg.
# Commands can be changed later.
FLAGS = flags.FLAGS
_SERVER_ADDRESS = flags.DEFINE_string('server', 'dns:///[::1]:10000',
                                      'server address')
_PIPELINE_NAME = flags.DEFINE_string('name', 'test-ImportSchemaGen2',
                                     'pipeline name')
_STORAGE_BUCKET = flags.DEFINE_string('bucket', '', 'storage bucket')


def main(unused_argv):
  prefix = f'gs://{_STORAGE_BUCKET.value}'
  sample_pipeline = pipeline.Pipeline(
      pipeline_name=_PIPELINE_NAME.value,
      pipeline_root=prefix + '/tfx/pipelines',
      components=[
          tfx.components.ImportSchemaGen(prefix + '/data/schema.pbtxt')
      ],
      enable_cache=False)
  pipeline_ir = compiler.Compiler().compile(sample_pipeline)
  runtime_parameter_utils.substitute_runtime_parameter(
      pipeline_ir, {
          constants.PIPELINE_RUN_ID_PARAMETER_NAME:
              datetime.datetime.now().isoformat(),
      })

  channel_creds = grpc.local_channel_credentials()
  with grpc.secure_channel(_SERVER_ADDRESS.value, channel_creds) as channel:
    grpc.channel_ready_future(channel).result()
    stub = service_pb2_grpc.KubernetesOrchestratorStub(channel)
    request = service_pb2.StartPipelineRequest(pipeline=pipeline_ir)
    stub.StartPipeline(request)


if __name__ == '__main__':
  app.run(main)
