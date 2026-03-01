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
"""Mock utilities for TFX pipeline IR."""

from typing import List

from tfx.dsl.components.base import executor_spec
from tfx.experimental.pipeline_testing import base_stub_executor
from tfx.proto.orchestration import executable_spec_pb2
from tfx.proto.orchestration import pipeline_pb2


def replace_executor_with_stub(pipeline: pipeline_pb2.Pipeline,
                               test_data_dir: str,
                               test_component_ids: List[str]):
  """Replace executors in pipeline IR with the stub executor.

  This funciton will replace the IR inplace.
  For example,

  pipeline_mock.replace_executor_with_stub(
      pipeline_ir,
      test_data_dir,
      test_component_ids = ['Trainer', 'Transform'])

  Then you can pass the modified `pipeline_ir` into a dag runner to execute
  the stubbed pipeline.

  Args:
    pipeline: The pipeline to alter.
    test_data_dir: The directory where pipeline outputs are recorded
      (pipeline_recorder.py).
    test_component_ids: List of ids of components that are to be tested. In
      other words, executors of components other than those specified by this
      list will be replaced with a BaseStubExecutor.

  Returns:
    None
  """
  deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
  if not pipeline.deployment_config.Unpack(deployment_config):
    raise NotImplementedError(
        'Unexpected pipeline.deployment_config type "{}". Currently only '
        'IntermediateDeploymentConfig is supported.'.format(
            pipeline.deployment_config.type_url))

  for component_id in deployment_config.executor_specs:
    if component_id not in test_component_ids:
      executable_spec = deployment_config.executor_specs[component_id]
      if executable_spec.Is(
          executable_spec_pb2.PythonClassExecutableSpec.DESCRIPTOR):
        stub_executor_class_spec = executor_spec.ExecutorClassSpec(
            base_stub_executor.BaseStubExecutor)
        stub_executor_class_spec.add_extra_flags(
            (base_stub_executor.TEST_DATA_DIR_FLAG + '=' + test_data_dir,
             base_stub_executor.COMPONENT_ID_FLAG + '=' + component_id))
        stub_executor_spec = stub_executor_class_spec.encode()
        executable_spec.Pack(stub_executor_spec)
      elif executable_spec.Is(
          executable_spec_pb2.BeamExecutableSpec.DESCRIPTOR):
        stub_beam_executor_spec = executor_spec.BeamExecutorSpec(
            base_stub_executor.BaseStubExecutor)
        stub_beam_executor_spec.add_extra_flags(
            (base_stub_executor.TEST_DATA_DIR_FLAG + '=' + test_data_dir,
             base_stub_executor.COMPONENT_ID_FLAG + '=' + component_id))
        stub_executor_spec = stub_beam_executor_spec.encode()
        executable_spec.Pack(stub_executor_spec)
      else:
        raise NotImplementedError(
            'Unexpected executable_spec type "{}". Currently only '
            'PythonClassExecutableSpec and BeamExecutorSpec is supported.'
            .format(executable_spec.type_url))
  pipeline.deployment_config.Pack(deployment_config)
