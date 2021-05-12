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
"""Test utilities for kubeflow v2 runner."""

import os
import subprocess
from typing import List, Text

from absl import logging
from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow_model_analysis as tfma
from tfx import components
from tfx import types
from tfx.components.trainer.executor import Executor
from tfx.dsl.component.experimental import container_component
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import importer
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies import latest_artifact_strategy
from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_big_query.example_gen import component as big_query_example_gen_component
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import test_utils
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.experimental import simple_artifacts
from tfx.utils import io_utils
from tfx.utils import test_case_utils

from google.protobuf import message

_TEST_TWO_STEP_PIPELINE_NAME = 'two-step-pipeline'

_TEST_FULL_PIPELINE_NAME = 'full-taxi-pipeline'

_TEST_PIPELINE_ROOT = 'path/to/my/root'

_TEST_INPUT_DATA = 'path/to/my/data'

_TEST_MODULE_FILE_LOCATION = 'path/to/my/module_utils.py'

TEST_RUNTIME_CONFIG = pipeline_pb2.PipelineJob.RuntimeConfig(
    gcs_output_directory=_TEST_PIPELINE_ROOT,
    parameters={
        'string_param': pipeline_pb2.Value(string_value='test-string'),
        'int_param': pipeline_pb2.Value(int_value=42),
        'float_param': pipeline_pb2.Value(double_value=3.14)
    })

_POLLING_INTERVAL_IN_SECONDS = 60

_MAX_JOB_EXECUTION_TIME_IN_SECONDS = 2400

_KUBEFLOW_SUCCEEDED_STATE = 'SUCCEEDED'

_KUBEFLOW_RUNNING_STATES = frozenset(('PENDING', 'RUNNING'))


# TODO(b/158245564): Reevaluate whether to keep this test helper function
def two_step_pipeline() -> tfx_pipeline.Pipeline:
  """Returns a simple 2-step pipeline under test."""
  example_gen = big_query_example_gen_component.BigQueryExampleGen(
      query='SELECT * FROM TABLE')
  statistics_gen = components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  return tfx_pipeline.Pipeline(
      pipeline_name=_TEST_TWO_STEP_PIPELINE_NAME,
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[example_gen, statistics_gen],
      # Needs to set GCP project because BQ is used.
      beam_pipeline_args=[
          '--project=my-gcp-project',
      ])


def create_pipeline_components(
    pipeline_root: Text,
    transform_module: Text,
    trainer_module: Text,
    bigquery_query: Text = '',
    csv_input_location: Text = '',
) -> List[base_node.BaseNode]:
  """Creates components for a simple Chicago Taxi TFX pipeline for testing.

  Args:
    pipeline_root: The root of the pipeline output.
    transform_module: The location of the transform module file.
    trainer_module: The location of the trainer module file.
    bigquery_query: The query to get input data from BigQuery. If not empty,
      BigQueryExampleGen will be used.
    csv_input_location: The location of the input data directory.

  Returns:
    A list of TFX components that constitutes an end-to-end test pipeline.
  """

  if bool(bigquery_query) == bool(csv_input_location):
    raise ValueError(
        'Exactly one example gen is expected. ',
        'Please provide either bigquery_query or csv_input_location.')

  if bigquery_query:
    example_gen = big_query_example_gen_component.BigQueryExampleGen(
        query=bigquery_query)
  else:
    example_gen = components.CsvExampleGen(input_base=csv_input_location)

  statistics_gen = components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  schema_gen = components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  example_validator = components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  transform = components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module)
  latest_model_resolver = resolver.Resolver(
      strategy_class=latest_artifact_strategy.LatestArtifactStrategy,
      model=channel.Channel(type=standard_artifacts.Model)).with_id(
          'Resolver.latest_model_resolver')
  trainer = components.Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(Executor),
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      base_model=latest_model_resolver.outputs['model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10),
      eval_args=trainer_pb2.EvalArgs(num_steps=5),
      module_file=trainer_module,
  )
  # Get the latest blessed model for model validation.
  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_strategy.LatestBlessedModelStrategy,
      model=channel.Channel(type=standard_artifacts.Model),
      model_blessing=channel.Channel(type=standard_artifacts.ModelBlessing)
  ).with_id('Resolver.latest_blessed_model_resolver')
  # Set the TFMA config for Model Evaluation and Validation.
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      metrics_specs=[
          tfma.MetricsSpec(
              metrics=[tfma.MetricConfig(class_name='ExampleCount')],
              thresholds={
                  'binary_accuracy':
                      tfma.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.5}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ])
  evaluator = components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  pusher = components.Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, schema_gen, example_validator, transform,
      latest_model_resolver, trainer, model_resolver, evaluator, pusher
  ]


# TODO(b/158245564): Reevaluate whether to keep this test helper function
def full_taxi_pipeline() -> tfx_pipeline.Pipeline:
  """Returns a full taxi pipeline under test."""
  pipeline_components = create_pipeline_components(
      pipeline_root=_TEST_PIPELINE_ROOT,
      transform_module=_TEST_MODULE_FILE_LOCATION,
      trainer_module=_TEST_MODULE_FILE_LOCATION,
      csv_input_location=_TEST_INPUT_DATA)

  return tfx_pipeline.Pipeline(
      pipeline_name=_TEST_FULL_PIPELINE_NAME,
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=pipeline_components)


class TransformerSpec(types.ComponentSpec):
  INPUTS = {
      'input1': component_spec.ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
      'output1': component_spec.ChannelParameter(type=standard_artifacts.Model),
  }
  PARAMETERS = {
      'param1': component_spec.ExecutionParameter(type=Text),
  }


class ProducerSpec(types.ComponentSpec):
  INPUTS = {}
  OUTPUTS = {
      'output1': component_spec.ChannelParameter(type=standard_artifacts.Model),
  }
  PARAMETERS = {
      'param1': component_spec.ExecutionParameter(type=Text),
  }


class DummyContainerSpecComponent(base_component.BaseComponent):
  """Dummy ContainerSpec component."""

  SPEC_CLASS = TransformerSpec
  EXECUTOR_SPEC = executor_specs.TemplatedExecutorContainerSpec(
      image='dummy/transformer',
      command=[
          'transformer',
          '--input1',
          placeholders.InputUriPlaceholder('input1'),
          '--output1',
          placeholders.OutputUriPlaceholder('output1'),
          '--param1',
          placeholders.InputValuePlaceholder('param1'),
      ])

  def __init__(self, input1, param1, output1, instance_name=None):
    spec = TransformerSpec(
        input1=input1,
        output1=output1,
        param1=param1,
    )
    super(DummyContainerSpecComponent, self).__init__(spec=spec)
    if instance_name:
      self._id = '{}.{}'.format(self.__class__.__name__, instance_name)
    else:
      self._id = self.__class__.__name__


class DummyProducerComponent(base_component.BaseComponent):
  """Dummy producer component."""

  SPEC_CLASS = ProducerSpec
  EXECUTOR_SPEC = executor_specs.TemplatedExecutorContainerSpec(
      image='dummy/producer',
      command=[
          'producer',
          '--output1',
          placeholders.OutputUriPlaceholder('output1'),
          '--param1',
          placeholders.InputValuePlaceholder('param1'),
          '--wrapped-param',
          placeholders.ConcatPlaceholder([
              'prefix-',
              placeholders.InputValuePlaceholder('param1'),
              '-suffix',
          ]),
      ])

  def __init__(self, param1, output1, instance_name=None):
    spec = ProducerSpec(
        output1=output1,
        param1=param1,
    )
    super(DummyProducerComponent, self).__init__(spec=spec)
    if instance_name:
      self._id = '{}.{}'.format(self.__class__.__name__, instance_name)
    else:
      self._id = self.__class__.__name__


dummy_transformer_component = container_component.create_container_component(
    name='DummyContainerSpecComponent',
    inputs={
        'input1': standard_artifacts.Model,
    },
    outputs={
        'output1': standard_artifacts.Model,
    },
    parameters={
        'param1': str,
    },
    image='dummy/transformer',
    command=[
        'transformer',
        '--input1',
        placeholders.InputUriPlaceholder('input1'),
        '--output1',
        placeholders.OutputUriPlaceholder('output1'),
        '--param1',
        placeholders.InputValuePlaceholder('param1'),
    ],
)

dummy_producer_component = container_component.create_container_component(
    name='DummyProducerComponent',
    outputs={
        'output1': standard_artifacts.Model,
    },
    parameters={
        'param1': str,
    },
    image='dummy/producer',
    command=[
        'producer',
        '--output1',
        placeholders.OutputUriPlaceholder('output1'),
        '--param1',
        placeholders.InputValuePlaceholder('param1'),
        '--wrapped-param',
        placeholders.ConcatPlaceholder([
            'prefix-',
            placeholders.InputValuePlaceholder('param1'),
            '-suffix',
        ]),
    ],
)


def pipeline_with_one_container_spec_component() -> tfx_pipeline.Pipeline:
  """Pipeline with container."""

  importer_task = importer.Importer(
      source_uri='some-uri',
      artifact_type=standard_artifacts.Model,
  ).with_id('my_importer')

  container_task = DummyContainerSpecComponent(
      input1=importer_task.outputs['result'],
      output1=channel_utils.as_channel([standard_artifacts.Model()]),
      param1='value1',
  )

  return tfx_pipeline.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[importer_task, container_task],
  )


def pipeline_with_two_container_spec_components() -> tfx_pipeline.Pipeline:
  """Pipeline with container."""

  container1_task = DummyProducerComponent(
      output1=channel_utils.as_channel([standard_artifacts.Model()]),
      param1='value1',
  )

  container2_task = DummyContainerSpecComponent(
      input1=container1_task.outputs['output1'],
      output1=channel_utils.as_channel([standard_artifacts.Model()]),
      param1='value2',
  )

  return tfx_pipeline.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[container1_task, container2_task],
  )


def pipeline_with_two_container_spec_components_2() -> tfx_pipeline.Pipeline:
  """Pipeline with container."""

  container1_task = dummy_producer_component(
      output1=channel_utils.as_channel([standard_artifacts.Model()]),
      param1='value1',
  )

  container2_task = dummy_transformer_component(
      input1=container1_task.outputs['output1'],
      output1=channel_utils.as_channel([standard_artifacts.Model()]),
      param1='value2',
  )

  return tfx_pipeline.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[container1_task, container2_task],
  )


def get_proto_from_test_data(filename: Text,
                             pb_message: message.Message) -> message.Message:
  """Helper function that gets proto from testdata."""
  filepath = os.path.join(os.path.dirname(__file__), 'testdata', filename)
  return io_utils.parse_pbtxt_file(filepath, pb_message)


def get_text_from_test_data(filename: Text) -> Text:
  """Helper function that gets raw string from testdata."""
  filepath = os.path.join(os.path.dirname(__file__), 'testdata', filename)
  return fileio.open(filepath, 'rb').read().decode('utf-8')


class _ProducerComponentSpec(component_spec.ComponentSpec):
  """Test component spec using AI Platform simple artifact types."""
  INPUTS = {}
  OUTPUTS = {
      'examples':
          component_spec.ChannelParameter(type=simple_artifacts.Dataset),
      'external_data':
          component_spec.ChannelParameter(type=simple_artifacts.File),
  }
  PARAMETERS = {}


class _ConsumerComponentSpec(component_spec.ComponentSpec):
  """Test component spec using AI Platform simple artifact types."""

  INPUTS = {
      'examples':
          component_spec.ChannelParameter(type=simple_artifacts.Dataset),
      'external_data':
          component_spec.ChannelParameter(type=simple_artifacts.File),
  }
  OUTPUTS = {
      'stats':
          component_spec.ChannelParameter(type=simple_artifacts.Statistics),
      'metrics':
          component_spec.ChannelParameter(type=simple_artifacts.Metrics)
  }
  PARAMETERS = {}


class ProducerComponent(base_component.BaseComponent):
  """Test component used in step 1 of a 2-step pipeline testing AI Platform simple artifact types."""
  SPEC_CLASS = _ProducerComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor_class=base_executor.EmptyExecutor)

  def __init__(self):
    examples_channel = channel_utils.as_channel([simple_artifacts.Dataset()])
    external_data_channel = channel_utils.as_channel([simple_artifacts.File()])
    super(ProducerComponent, self).__init__(
        _ProducerComponentSpec(
            examples=examples_channel, external_data=external_data_channel))


class ConsumerComponent(base_component.BaseComponent):
  """Test component used in step 2 of a 2-step pipeline testing AI Platform simple artifact types."""
  SPEC_CLASS = _ConsumerComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor_class=base_executor.EmptyExecutor)

  def __init__(self, examples: channel.Channel, external_data: channel.Channel):
    stats_output_channel = channel_utils.as_channel(
        [simple_artifacts.Statistics()])
    metrics_output_channel = channel_utils.as_channel(
        [simple_artifacts.Metrics()])
    super(ConsumerComponent, self).__init__(
        _ConsumerComponentSpec(
            examples=examples,
            external_data=external_data,
            stats=stats_output_channel,
            metrics=metrics_output_channel))


def two_step_kubeflow_artifacts_pipeline() -> tfx_pipeline.Pipeline:
  """Builds 2-Step pipeline to test AI Platform simple artifact types."""
  step1 = ProducerComponent()
  step2 = ConsumerComponent(
      examples=step1.outputs['examples'],
      external_data=step1.outputs['external_data'])
  return tfx_pipeline.Pipeline(
      pipeline_name='two-step-kubeflow-artifacts-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[step1, step2],
      beam_pipeline_args=[
          '--project=my-gcp-project',
      ])


def two_step_pipeline_with_task_only_dependency() -> tfx_pipeline.Pipeline:
  """Returns a simple 2-step pipeline with task only dependency between them."""

  step_1 = container_component.create_container_component(
      name='Step 1',
      inputs={},
      outputs={},
      parameters={},
      image='step-1-image',
      command=['run', 'step-1'])()

  step_2 = container_component.create_container_component(
      name='Step 2',
      inputs={},
      outputs={},
      parameters={},
      image='step-2-image',
      command=['run', 'step-2'])()
  step_2.add_upstream_node(step_1)

  return tfx_pipeline.Pipeline(
      pipeline_name='two-step-task-only-dependency-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[step_1, step_2],
  )


primitive_producer_component = container_component.create_container_component(
    name='ProducePrimitives',
    outputs={
        'output_string': standard_artifacts.String,
        'output_int': standard_artifacts.Integer,
        'output_float': standard_artifacts.Float,
    },
    image='busybox',
    command=[
        'produce',
        placeholders.OutputUriPlaceholder('output_string'),
        placeholders.OutputUriPlaceholder('output_int'),
        placeholders.OutputUriPlaceholder('output_float'),
    ],
)

primitive_consumer_component = container_component.create_container_component(
    name='ConsumeByValue',
    inputs={
        'input_string': standard_artifacts.String,
        'input_int': standard_artifacts.Integer,
        'input_float': standard_artifacts.Float,
    },
    parameters={
        'param_string': str,
        'param_int': int,
        'param_float': float,
    },
    image='busybox',
    command=[
        'consume',
        placeholders.InputValuePlaceholder('input_string'),
        placeholders.InputValuePlaceholder('input_int'),
        placeholders.InputValuePlaceholder('input_float'),
        placeholders.InputValuePlaceholder('param_string'),
        placeholders.InputValuePlaceholder('param_int'),
        placeholders.InputValuePlaceholder('param_float'),
    ],
)


def consume_primitive_artifacts_by_value_pipeline() -> tfx_pipeline.Pipeline:
  """Pipeline which features consuming artifacts by value."""

  producer_task = primitive_producer_component()

  consumer_task = primitive_consumer_component(
      input_string=producer_task.outputs['output_string'],
      input_int=producer_task.outputs['output_int'],
      input_float=producer_task.outputs['output_float'],
      param_string='string value',
      param_int=42,
      param_float=3.14,
  )

  return tfx_pipeline.Pipeline(
      pipeline_name='consume-primitive-artifacts-by-value-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[producer_task, consumer_task],
  )


def pipeline_with_runtime_parameter() -> tfx_pipeline.Pipeline:
  """Pipeline which contains a runtime parameter."""

  producer_task = primitive_producer_component()

  consumer_task = primitive_consumer_component(
      input_string=producer_task.outputs['output_string'],
      input_int=producer_task.outputs['output_int'],
      input_float=producer_task.outputs['output_float'],
      param_string=data_types.RuntimeParameter(
          ptype=Text, name='string_param', default='string value'),
      param_int=42,
      param_float=3.14,
  )

  return tfx_pipeline.Pipeline(
      pipeline_name='pipeline-with-runtime-parameter',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[producer_task, consumer_task],
  )


def tasks_for_pipeline_with_artifact_value_passing():
  """A simple pipeline with artifact consumed as value."""
  producer_component = container_component.create_container_component(
      name='Produce',
      outputs={
          'data': simple_artifacts.File,
      },
      parameters={
          'message': str,
      },
      image='gcr.io/ml-pipeline/mirrors/cloud-sdk',
      command=[
          'sh',
          '-exc',
          """
            message="$0"
            output_data_uri="$1"
            output_data_path=$(mktemp)

            # Running the main code
            echo "Hello $message" >"$output_data_path"

            # Getting data out of the container
            gsutil cp -r "$output_data_path" "$output_data_uri"
          """,
          placeholders.InputValuePlaceholder('message'),
          placeholders.OutputUriPlaceholder('data'),
      ],
  )

  print_value_component = container_component.create_container_component(
      name='Print',
      inputs={
          'text': simple_artifacts.File,
      },
      image='gcr.io/ml-pipeline/mirrors/cloud-sdk',
      command=[
          'echo',
          placeholders.InputValuePlaceholder('text'),
      ],
  )

  producer_task = producer_component(message='World!')
  print_task = print_value_component(text=producer_task.outputs['data'],)
  return [producer_task, print_task]


class BaseKubeflowV2Test(test_case_utils.TfxTest):
  """Defines testing harness for pipeline on KubeflowV2DagRunner."""

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The src path to use to build docker image
  _REPO_BASE = os.environ.get('KFP_E2E_SRC')

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ.get('KFP_E2E_BASE_CONTAINER_IMAGE')

  # The project id to use to run tests.
  _GCP_PROJECT_ID = os.environ.get('KFP_E2E_GCP_PROJECT_ID')

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ.get('KFP_E2E_BUCKET_NAME')

  # The location of test user module file.
  # It is retrieved from inside the container subject to testing.
  # This location depends on install path of TFX in the docker image.
  _MODULE_FILE = '/opt/conda/lib/python3.7/site-packages/tfx/examples/chicago_taxi_pipeline/taxi_utils.py'

  _CONTAINER_IMAGE = '{}:{}'.format(_BASE_CONTAINER_IMAGE,
                                    test_utils.random_id())

  @classmethod
  def setUpClass(cls):
    super(BaseKubeflowV2Test, cls).setUpClass()

    # Create a container image for use by test pipelines.
    test_utils.build_and_push_docker_image(cls._CONTAINER_IMAGE, cls._REPO_BASE)

  @classmethod
  def tearDownClass(cls):
    super(BaseKubeflowV2Test, cls).tearDownClass()

    # Delete container image used in tests.
    logging.info('Deleting image %s', cls._CONTAINER_IMAGE)
    subprocess.run(
        ['gcloud', 'container', 'images', 'delete', cls._CONTAINER_IMAGE],
        check=True)

  def setUp(self):
    super(BaseKubeflowV2Test, self).setUp()
    self.enter_context(test_case_utils.change_working_dir(self.tmp_dir))

    self._test_dir = self.tmp_dir
    self._test_output_dir = 'gs://{}/test_output'.format(self._BUCKET_NAME)

  def _pipeline_root(self, pipeline_name: Text):
    return os.path.join(self._test_output_dir, pipeline_name)

  def _delete_pipeline_output(self, pipeline_name: Text):
    """Deletes output produced by the named pipeline.

    Args:
      pipeline_name: The name of the pipeline.
    """
    test_utils.delete_gcs_files(self._GCP_PROJECT_ID, self._BUCKET_NAME,
                                'test_output/{}'.format(pipeline_name))

  def _create_pipeline(
      self,
      pipeline_name: Text,
      pipeline_components: List[base_node.BaseNode],
      beam_pipeline_args: List[Text] = None) -> tfx_pipeline.Pipeline:
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args)
