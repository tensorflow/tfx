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
"""Tests for tfx.orchestration.managed.pipeline_builder."""

from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow as tf
from tfx.orchestration.kubeflow import decorators
from tfx.orchestration.kubeflow.v2 import pipeline_builder
from tfx.orchestration.kubeflow.v2 import test_utils

_VALID_NAME = 'this-name-is-good'
_BAD_NAME = 'This  is  not  a GOOD name.'


class PipelineBuilderTest(tf.test.TestCase):

  def testCheckName(self):
    # Should pass the check with the legal name.
    pipeline_builder._check_name(_VALID_NAME)
    # Should fail the check with the illegal name.
    with self.assertRaisesRegex(ValueError, 'User provided pipeline name'):
      pipeline_builder._check_name(_BAD_NAME)

  def testBuildTwoStepPipeline(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_pipeline(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data('expected_two_step_pipeline.pbtxt',
                                            pipeline_pb2.PipelineSpec()),
        actual_pipeline_spec)

  def testBuildRuntimeConfig(self):
    my_builder = pipeline_builder.RuntimeConfigBuilder(
        pipeline_info=test_utils.two_step_pipeline().pipeline_info,
        parameter_values={
            'string_param': 'test-string',
            'int_param': 42,
            'float_param': 3.14
        })
    actual_output_path_config = my_builder.build()
    self.assertProtoEquals(test_utils.TEST_RUNTIME_CONFIG,
                           actual_output_path_config)

  def testBuildPipelineWithOneContainerSpecComponent(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_one_container_spec_component(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_one_container_spec_component.pbtxt',
            pipeline_pb2.PipelineSpec()), actual_pipeline_spec)

  def testBuildPipelineWithTwoContainerSpecComponents(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_two_container_spec_components(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_two_container_spec_components.pbtxt',
            pipeline_pb2.PipelineSpec()), actual_pipeline_spec)

  def testBuildPipelineWithTwoContainerSpecComponents2(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_two_container_spec_components_2(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            # Same as in testBuildPipelineWithTwoContainerSpecComponents
            'expected_pipeline_with_two_container_spec_components.pbtxt',
            pipeline_pb2.PipelineSpec()),
        actual_pipeline_spec)

  def testBuildPipelineWithPrimitiveValuePassing(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.consume_primitive_artifacts_by_value_pipeline(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_consume_primitive_artifacts_by_value_pipeline.pbtxt',
            pipeline_pb2.PipelineSpec()), actual_pipeline_spec)

  def testBuildPipelineWithRuntimeParameter(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_runtime_parameter(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_runtime_parameter.pbtxt',
            pipeline_pb2.PipelineSpec()), actual_pipeline_spec)

  def testKubeflowArtifactsTwoStepPipeline(self):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_kubeflow_artifacts_pipeline(),
        default_image='gcr.io/my-tfx:latest')
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_kubeflow_artifacts_pipeline.pbtxt',
            pipeline_pb2.PipelineSpec()), actual_pipeline_spec)

  def testTwoStepPipelineWithTaskOnlyDependency(self):
    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_pipeline_with_task_only_dependency(),
        default_image='unused-image')

    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_task_only_dependency.pbtxt',
            pipeline_pb2.PipelineSpec()), pipeline_spec)

  def testBuildTwoStepPipelineWithCacheEnabled(self):
    pipeline = test_utils.two_step_pipeline()
    pipeline.enable_cache = True

    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline, default_image='gcr.io/my-tfx:latest')
    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_cache_enabled.pbtxt',
            pipeline_pb2.PipelineSpec()), pipeline_spec)

  def testPipelineWithExitHandler(self):
    pipeline = test_utils.two_step_pipeline()
    # define exit handler
    exit_handler = test_utils.dummy_exit_handler(
        param1=decorators.FinalStatusStr())

    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline,
        default_image='gcr.io/my-tfx:latest',
        exit_handler=exit_handler)
    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_exit_handler.pbtxt',
            pipeline_pb2.PipelineSpec()), pipeline_spec)

  def testTwoStepPipelineWithDynamicExecutionProperties(self):
    pipeline = test_utils.two_step_pipeline_with_dynamic_exec_properties()
    pipeline_spec = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline, default_image='gcr.io/my-tfx:latest').build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_dynamic_execution_properties.pbtxt',
            pipeline_pb2.PipelineSpec()), pipeline_spec)


if __name__ == '__main__':
  tf.test.main()
