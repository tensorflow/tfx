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
"""Tests for tfx.experimental.pipeline_testing.pipeline_mock."""

import tensorflow as tf
from tfx.experimental.pipeline_testing import pipeline_mock
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import text_format


class PipelineMockTest(tf.test.TestCase):

  def testReplacePythonClassExecutorWithStub(self):
    pipeline = text_format.Parse(
        """
        deployment_config {
          [type.googleapis.com/tfx.orchestration.IntermediateDeploymentConfig] {
            executor_specs {
              key: "CsvExampleGen"
              value {
                [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
                  class_path: "tfx.components.example_gen.csv_example_gen.executor.Executor"
                  extra_flags: "--my_testing_extra_flags=foo"
                }
              }
            }
          }
        }""", pipeline_pb2.Pipeline())
    expected = """
        deployment_config {
          [type.googleapis.com/tfx.orchestration.IntermediateDeploymentConfig] {
            executor_specs {
              key: "CsvExampleGen"
              value {
                [type.googleapis.com/tfx.orchestration.executable_spec.PythonClassExecutableSpec] {
                  class_path: "tfx.experimental.pipeline_testing.base_stub_executor.BaseStubExecutor"
                  extra_flags: "--test_data_dir=/mock/a"
                  extra_flags: "--component_id=CsvExampleGen"
                }
              }
            }
          }
        }"""
    pipeline_mock.replace_executor_with_stub(pipeline, '/mock/a', [])
    self.assertProtoEquals(expected, pipeline)

  def testReplaceBeamExecutorWithStub(self):
    pipeline = text_format.Parse(
        """
        deployment_config {
          [type.googleapis.com/tfx.orchestration.IntermediateDeploymentConfig] {
            executor_specs {
              key: "CsvExampleGen"
              value {
                [type.googleapis.com/tfx.orchestration.executable_spec.BeamExecutableSpec] {
                  python_executor_spec {
                    class_path: "tfx.components.example_gen.csv_example_gen.executor.Executor"
                    extra_flags: "--my_testing_extra_flags=foo"
                  }
                }
              }
            }
          }
        }""", pipeline_pb2.Pipeline())
    expected = """
        deployment_config {
          [type.googleapis.com/tfx.orchestration.IntermediateDeploymentConfig] {
            executor_specs {
              key: "CsvExampleGen"
              value {
                [type.googleapis.com/tfx.orchestration.executable_spec.BeamExecutableSpec] {
                  python_executor_spec {
                    class_path: "tfx.experimental.pipeline_testing.base_stub_executor.BaseStubExecutor"
                    extra_flags: "--test_data_dir=/mock/a"
                    extra_flags: "--component_id=CsvExampleGen"
                  }
                }
              }
            }
          }
        }"""
    pipeline_mock.replace_executor_with_stub(pipeline, '/mock/a', [])
    self.assertProtoEquals(expected, pipeline)


if __name__ == '__main__':
  tf.test.main()
