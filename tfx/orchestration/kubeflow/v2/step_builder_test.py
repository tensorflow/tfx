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
"""Tests for Kubeflow V2 step builder."""

from typing import Sequence

import tensorflow as tf
from tfx import components
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import step_builder
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.orchestration.kubeflow.v2.proto import pipeline_pb2
from tfx.proto import example_gen_pb2
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts

from google.protobuf import text_format

_TEST_CMDS = ('python', '-m', 'my_entrypoint.app_module')

_EXPECTED_LATEST_BLESSED_MODEL_RESOLVER_1 = r"""
task_info {
  name: "Resolver.my_resolver2-model-blessing-resolver"
}
outputs {
  artifacts {
    key: "model_blessing"
    value {
      artifact_type {
        instance_schema: "title: tfx.ModelBlessing\ntype: object\nproperties:\n"
      }
    }
  }
}
executor_label: "Resolver.my_resolver2-model-blessing-resolver_executor"
"""

_EXPECTED_LATEST_BLESSED_MODEL_RESOLVER_2 = r"""
task_info {
  name: "Resolver.my_resolver2-model-resolver"
}
inputs {
  artifacts {
    key: "input"
    value {
      producer_task: "Resolver.my_resolver2-model-blessing-resolver"
      output_artifact_key: "model_blessing"
    }
  }
}
outputs {
  artifacts {
    key: "model"
    value {
      artifact_type {
        instance_schema: "title: tfx.Model\ntype: object\nproperties:\n"
      }
    }
  }
}
executor_label: "Resolver.my_resolver2-model-resolver_executor"
"""

_EXPECTED_LATEST_BLESSED_MODEL_EXECUTOR = r"""
executors {
  key: "Resolver.my_resolver2-model-blessing-resolver_executor"
  value {
    resolver {
      output_artifact_queries {
        key: "model_blessing"
        value {
          filter: "artifact_type='tfx.ModelBlessing' and state=LIVE and custom_properties['blessed']='1'"
        }
      }
    }
  }
}
executors {
  key: "Resolver.my_resolver2-model-resolver_executor"
  value {
    resolver {
      output_artifact_queries {
        key: "model"
        value {
          filter: "artifact_type='tfx.Model' and state=LIVE and name={$.inputs.artifacts[\'input\'].custom_properties[\'current_model_id\']}"
        }
      }
    }
  }
}
"""


class StepBuilderTest(tf.test.TestCase):

  def _sole(
      self, task_specs: Sequence[pipeline_pb2.PipelineTaskSpec]
  ) -> pipeline_pb2.PipelineTaskSpec:
    """Asserts the task_specs has length 1 and returns the only element."""
    self.assertLen(task_specs, 1)
    return task_specs[0]

  def testBuildTask(self):
    query = 'SELECT * FROM TABLE'
    bq_example_gen = big_query_example_gen_component.BigQueryExampleGen(
        query=query)
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=bq_example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config,
        enable_cache=True)
    actual_step_spec = self._sole(my_builder.build())

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_bq_example_gen.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_bq_example_gen_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildContainerTask(self):
    task = test_utils.DummyProducerComponent(
        output1=channel_utils.as_channel([standard_artifacts.Model()]),
        param1='value1',
    )
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=task,
        image='gcr.io/tensorflow/tfx:latest',  # Note this has no effect here.
        deployment_config=deployment_config)
    actual_step_spec = self._sole(my_builder.build())

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_task.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildContainerTask2(self):
    task = test_utils.dummy_producer_component(
        output1=channel_utils.as_channel([standard_artifacts.Model()]),
        param1='value1',
    )
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=task,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config)
    actual_step_spec = self._sole(my_builder.build())

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            # Same as in testBuildContainerTask
            'expected_dummy_container_spec_task.pbtxt',
            pipeline_pb2.PipelineTaskSpec()),
        actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildFileBasedExampleGen(self):
    beam_pipeline_args = ['runner=DataflowRunner']
    example_gen = components.CsvExampleGen(input_base='path/to/data/root')
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        image_cmds=_TEST_CMDS,
        beam_pipeline_args=beam_pipeline_args,
        deployment_config=deployment_config)
    actual_step_spec = self._sole(my_builder.build())
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_csv_example_gen.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_csv_example_gen_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildFileBasedExampleGenWithInputConfig(self):
    input_config = example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='*train.tfr'),
        example_gen_pb2.Input.Split(name='eval', pattern='*test.tfr')
    ])
    example_gen = components.ImportExampleGen(
        input_base='path/to/data/root', input_config=input_config)
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config)
    actual_step_spec = self._sole(my_builder.build())
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_import_example_gen.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_import_example_gen_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildImporter(self):
    impt = importer.Importer(
        instance_name='my_importer',
        source_uri='m/y/u/r/i',
        properties={
            'split_names': '["train", "eval"]',
        },
        custom_properties={
            'str_custom_property': 'abc',
            'int_custom_property': 123,
        },
        artifact_type=standard_artifacts.Examples)
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=impt, deployment_config=deployment_config)
    actual_step_spec = self._sole(my_builder.build())
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_importer.pbtxt', pipeline_pb2.PipelineTaskSpec()),
        actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_importer_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildLatestBlessedModelResolverSucceed(self):

    latest_blessed_resolver = resolver.Resolver(
        instance_name='my_resolver2',
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=channel.Channel(type=standard_artifacts.Model),
        model_blessing=channel.Channel(type=standard_artifacts.ModelBlessing))
    test_pipeline_info = data_types.PipelineInfo(
        pipeline_name='test-pipeline', pipeline_root='gs://path/to/my/root')

    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    my_builder = step_builder.StepBuilder(
        node=latest_blessed_resolver,
        deployment_config=deployment_config,
        pipeline_info=test_pipeline_info)
    actual_step_specs = my_builder.build()

    self.assertProtoEquals(
        text_format.Parse(_EXPECTED_LATEST_BLESSED_MODEL_RESOLVER_1,
                          pipeline_pb2.PipelineTaskSpec()),
        actual_step_specs[0])

    self.assertProtoEquals(
        text_format.Parse(_EXPECTED_LATEST_BLESSED_MODEL_RESOLVER_2,
                          pipeline_pb2.PipelineTaskSpec()),
        actual_step_specs[1])

    self.assertProtoEquals(
        text_format.Parse(_EXPECTED_LATEST_BLESSED_MODEL_EXECUTOR,
                          pipeline_pb2.PipelineDeploymentConfig()),
        deployment_config)

  def testBuildLatestArtifactResolverSucceed(self):
    latest_model_resolver = resolver.Resolver(
        instance_name='my_resolver',
        strategy_class=latest_artifacts_resolver.LatestArtifactsResolver,
        model=channel.Channel(type=standard_artifacts.Model),
        examples=channel.Channel(type=standard_artifacts.Examples))
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    test_pipeline_info = data_types.PipelineInfo(
        pipeline_name='test-pipeline', pipeline_root='gs://path/to/my/root')
    my_builder = step_builder.StepBuilder(
        node=latest_model_resolver,
        deployment_config=deployment_config,
        pipeline_info=test_pipeline_info)
    actual_step_spec = self._sole(my_builder.build())

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_artifact_resolver.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_artifact_resolver_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)


if __name__ == '__main__':
  tf.test.main()
