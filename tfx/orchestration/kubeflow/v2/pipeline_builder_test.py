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

from absl.testing import parameterized
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow as tf
from tfx.orchestration.kubeflow import decorators
from tfx.orchestration.kubeflow.v2 import pipeline_builder
from tfx.orchestration.kubeflow.v2 import test_utils

_VALID_NAME = 'this-name-is-good'
_BAD_NAME = 'This  is  not  a GOOD name.'


class PipelineBuilderTest(tf.test.TestCase, parameterized.TestCase):

  def testCheckName(self):
    # Should pass the check with the legal name.
    pipeline_builder._check_name(_VALID_NAME)
    # Should fail the check with the illegal name.
    with self.assertRaisesRegex(ValueError, 'User provided pipeline name'):
      pipeline_builder._check_name(_BAD_NAME)

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildTwoStepPipeline(self, use_pipeline_spec_2_1):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_pipeline(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildTwoStepPipelineWithMultipleImages(self, use_pipeline_spec_2_1):
    images = {
        pipeline_builder.DEFAULT_IMAGE_PATH_KEY: 'gcr.io/my-tfx:latest',
        'BigQueryExampleGen': 'gcr.io/big-query:1.0.0',
    }
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_pipeline(),
        default_image=images,
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_multiple_images.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildRuntimeConfig(self, use_pipeline_spec_2_1):
    my_builder = pipeline_builder.RuntimeConfigBuilder(
        pipeline_info=test_utils.two_step_pipeline().pipeline_info,
        parameter_values={
            'string_param': 'test-string',
            'int_param': 42,
            'float_param': 3.14,
        },
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_output_path_config = my_builder.build()
    if use_pipeline_spec_2_1:
      self.assertProtoEquals(
          test_utils.TEST_RUNTIME_CONFIG, actual_output_path_config
      )
    else:
      self.assertProtoEquals(
          test_utils.TEST_RUNTIME_CONFIG_LEGACY, actual_output_path_config
      )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildPipelineWithOneContainerSpecComponent(
      self, use_pipeline_spec_2_1
  ):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_one_container_spec_component(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_one_container_spec_component.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildPipelineWithTwoContainerSpecComponents(
      self, use_pipeline_spec_2_1
  ):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_two_container_spec_components(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_two_container_spec_components.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildPipelineWithTwoContainerSpecComponents2(
      self, use_pipeline_spec_2_1
  ):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_two_container_spec_components_2(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            # Same as in testBuildPipelineWithTwoContainerSpecComponents
            'expected_pipeline_with_two_container_spec_components.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildPipelineWithPrimitiveValuePassing(self, use_pipeline_spec_2_1):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.consume_primitive_artifacts_by_value_pipeline(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_consume_primitive_artifacts_by_value_pipeline.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildPipelineWithRuntimeParameter(self, use_pipeline_spec_2_1):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.pipeline_with_runtime_parameter(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_pipeline_with_runtime_parameter.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testKubeflowArtifactsTwoStepPipeline(self, use_pipeline_spec_2_1):
    my_builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_kubeflow_artifacts_pipeline(),
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    actual_pipeline_spec = my_builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_kubeflow_artifacts_pipeline.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        actual_pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testTwoStepPipelineWithTaskOnlyDependency(self, use_pipeline_spec_2_1):
    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=test_utils.two_step_pipeline_with_task_only_dependency(),
        default_image='unused-image',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )

    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_task_only_dependency.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testBuildTwoStepPipelineWithCacheEnabled(self, use_pipeline_spec_2_1):
    pipeline = test_utils.two_step_pipeline()
    pipeline.enable_cache = True

    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline,
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_cache_enabled.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testPipelineWithExitHandler(self, use_pipeline_spec_2_1):
    pipeline = test_utils.two_step_pipeline()
    # define exit handler
    exit_handler = test_utils.dummy_exit_handler(
        param1=decorators.FinalStatusStr())

    builder = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline,
        default_image='gcr.io/my-tfx:latest',
        exit_handler=exit_handler,
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    )
    pipeline_spec = builder.build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_exit_handler.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testTwoStepPipelineWithDynamicExecutionProperties(
      self, use_pipeline_spec_2_1
  ):
    pipeline = test_utils.two_step_pipeline_with_dynamic_exec_properties()
    pipeline_spec = pipeline_builder.PipelineBuilder(
        tfx_pipeline=pipeline,
        default_image='gcr.io/my-tfx:latest',
        use_pipeline_spec_2_1=use_pipeline_spec_2_1,
    ).build()
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_two_step_pipeline_with_dynamic_execution_properties.pbtxt',
            pipeline_pb2.PipelineSpec(),
            use_legacy_data=not use_pipeline_spec_2_1,
        ),
        pipeline_spec,
    )

  @parameterized.named_parameters(
      dict(testcase_name='use_pipeline_spec_2_1', use_pipeline_spec_2_1=True),
      dict(testcase_name='use_pipeline_spec_2_0', use_pipeline_spec_2_1=False),
  )
  def testTwoStepPipelineWithIllegalDynamicExecutionProperty(
      self, use_pipeline_spec_2_1
  ):
    pipeline = test_utils.two_step_pipeline_with_illegal_dynamic_exec_property()
    with self.assertRaisesRegex(
        ValueError, 'Invalid placeholder for exec prop range_config.*'
    ):
      pipeline_builder.PipelineBuilder(
          tfx_pipeline=pipeline,
          default_image='gcr.io/my-tfx:latest',
          use_pipeline_spec_2_1=use_pipeline_spec_2_1,
      ).build()
