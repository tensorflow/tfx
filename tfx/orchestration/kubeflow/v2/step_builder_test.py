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

from typing import Any, Dict

from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow as tf
from tfx import components
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.orchestration import data_types
from tfx.orchestration.kubeflow.v2 import step_builder
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.proto import example_gen_pb2
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts

_TEST_CMDS = ('python', '-m', 'my_entrypoint.app_module')


class StepBuilderTest(tf.test.TestCase):

  def _sole(self, d: Dict[Any, Any]) -> Any:
    """Asserts the dictionary has length 1 and returns the only value."""
    self.assertLen(d, 1)
    return list(d.values())[0]

  def testBuildTask(self):
    query = 'SELECT * FROM TABLE'
    bq_example_gen = big_query_example_gen_component.BigQueryExampleGen(
        query=query)
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=bq_example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config,
        component_defs=component_defs,
        enable_cache=True)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_bq_example_gen_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_bq_example_gen_task.pbtxt',
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
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=task,
        image='gcr.io/tensorflow/tfx:latest',  # Note this has no effect here.
        deployment_config=deployment_config,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
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
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=task,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    # Same as in testBuildContainerTask
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_task.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_dummy_container_spec_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildFileBasedExampleGen(self):
    beam_pipeline_args = ['runner=DataflowRunner']
    example_gen = components.CsvExampleGen(input_base='path/to/data/root')
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        image_cmds=_TEST_CMDS,
        beam_pipeline_args=beam_pipeline_args,
        deployment_config=deployment_config,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_csv_example_gen_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_csv_example_gen_task.pbtxt',
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
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=example_gen,
        image='gcr.io/tensorflow/tfx:latest',
        deployment_config=deployment_config,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_import_example_gen_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_import_example_gen_task.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_import_example_gen_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildImporter(self):
    impt = importer.Importer(
        source_uri='m/y/u/r/i',
        properties={
            'split_names': '["train", "eval"]',
        },
        custom_properties={
            'str_custom_property': 'abc',
            'int_custom_property': 123,
        },
        artifact_type=standard_artifacts.Examples).with_id('my_importer')
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=impt,
        deployment_config=deployment_config,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data('expected_importer_component.pbtxt',
                                            pipeline_pb2.ComponentSpec()),
        actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data('expected_importer_task.pbtxt',
                                            pipeline_pb2.PipelineTaskSpec()),
        actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_importer_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildLatestBlessedModelStrategySucceed(self):
    latest_blessed_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_strategy.LatestBlessedModelStrategy,
        model=channel.Channel(type=standard_artifacts.Model),
        model_blessing=channel.Channel(
            type=standard_artifacts.ModelBlessing)).with_id('my_resolver2')
    test_pipeline_info = data_types.PipelineInfo(
        pipeline_name='test-pipeline', pipeline_root='gs://path/to/my/root')

    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    component_defs = {}
    my_builder = step_builder.StepBuilder(
        node=latest_blessed_resolver,
        deployment_config=deployment_config,
        pipeline_info=test_pipeline_info,
        component_defs=component_defs)
    actual_step_specs = my_builder.build()

    model_blessing_resolver_id = 'my_resolver2-model-blessing-resolver'
    model_resolver_id = 'my_resolver2-model-resolver'
    self.assertSameElements(actual_step_specs.keys(),
                            [model_blessing_resolver_id, model_resolver_id])

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_blessed_model_resolver_component_1.pbtxt',
            pipeline_pb2.ComponentSpec()),
        component_defs[model_blessing_resolver_id])

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_blessed_model_resolver_task_1.pbtxt',
            pipeline_pb2.PipelineTaskSpec()),
        actual_step_specs[model_blessing_resolver_id])

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_blessed_model_resolver_component_2.pbtxt',
            pipeline_pb2.ComponentSpec()), component_defs[model_resolver_id])

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_blessed_model_resolver_task_2.pbtxt',
            pipeline_pb2.PipelineTaskSpec()),
        actual_step_specs[model_resolver_id])

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_blessed_model_resolver_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)

  def testBuildLatestArtifactResolverSucceed(self):
    latest_model_resolver = resolver.Resolver(
        strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
        model=channel.Channel(type=standard_artifacts.Model),
        examples=channel.Channel(
            type=standard_artifacts.Examples)).with_id('my_resolver')
    deployment_config = pipeline_pb2.PipelineDeploymentConfig()
    component_defs = {}
    test_pipeline_info = data_types.PipelineInfo(
        pipeline_name='test-pipeline', pipeline_root='gs://path/to/my/root')
    my_builder = step_builder.StepBuilder(
        node=latest_model_resolver,
        deployment_config=deployment_config,
        pipeline_info=test_pipeline_info,
        component_defs=component_defs)
    actual_step_spec = self._sole(my_builder.build())
    actual_component_def = self._sole(component_defs)

    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_artifact_resolver_component.pbtxt',
            pipeline_pb2.ComponentSpec()), actual_component_def)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_artifact_resolver_task.pbtxt',
            pipeline_pb2.PipelineTaskSpec()), actual_step_spec)
    self.assertProtoEquals(
        test_utils.get_proto_from_test_data(
            'expected_latest_artifact_resolver_executor.pbtxt',
            pipeline_pb2.PipelineDeploymentConfig()), deployment_config)


if __name__ == '__main__':
  tf.test.main()
