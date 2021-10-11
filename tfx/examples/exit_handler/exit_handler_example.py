# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Example to use exit handler."""

from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
from tfx import v1 as tfx
from tfx.utils import proto_utils

GOOGLE_CLOUD_PROJECT = 'tfx-oss-testing'
PIPELINE_NAME = 'testing-real-handler'
PIPELINE_ROOT = 'gs://zhujudy-dev/testing-custom-components'
PIPELINE_IMAGE = 'gcr.io/tfx-oss-testing/testing-real-handler'
PIPELINE_DEFINITION_FILE = 'testing-real-handler_pipeline.json'


@tfx.dsl.components.exit_handler
def exit_handler_component(final_status: tfx.dsl.components.Parameter[str]):
  # parse the final status
  pipeline_task_status = pipeline_pb2.PipelineTaskFinalStatus()
  proto_utils.json_to_proto(final_status, pipeline_task_status)
  print(pipeline_task_status)


@tfx.dsl.components.component
def hello_world_component():
  pass


@tfx.dsl.components.component
def hi_world_component():
  pass


def run_exit_handler_pipeline():
  """Implement to run a pipeline with exit handler."""
  hello_world = hello_world_component()
  hi_world = hi_world_component()
  exit_handler = exit_handler_component(
      final_status=tfx.dsl.experimental.FinalStatusStr())

  dsl_pipeline = tfx.dsl.Pipeline(
      pipeline_name=PIPELINE_NAME,
      pipeline_root=PIPELINE_ROOT,
      components=[hello_world, hi_world])

  runner_config = tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(
      default_image=PIPELINE_IMAGE)

  runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
      config=runner_config, output_filename=PIPELINE_DEFINITION_FILE)
  runner.set_exit_handler([exit_handler])
  runner.run(pipeline=dsl_pipeline)

# To compile the pipeline:
# python exit_handler_e2e_test.py
if __name__ == '__main__':
  run_exit_handler_pipeline()
