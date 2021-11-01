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
from typing import List

from kfp.pipeline_spec import pipeline_spec_pb2 as pipeline_pb2
import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.components.trainer.executor import Executor
from tfx.dsl.component.experimental import executor_specs
from tfx.dsl.component.experimental import placeholders
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental.conditionals import conditional
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types.experimental import simple_artifacts

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


# TODO(b/158245564): Reevaluate whether to keep this test helper function
def two_step_pipeline() -> tfx.dsl.Pipeline:
  """Returns a simple 2-step pipeline under test."""
  example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
      query='SELECT * FROM TABLE').with_beam_pipeline_args([
          '--runner=DataflowRunner',
      ])
  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  return tfx.dsl.Pipeline(
      pipeline_name=_TEST_TWO_STEP_PIPELINE_NAME,
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[example_gen, statistics_gen],
      # Needs to set GCP project because BQ is used.
      beam_pipeline_args=[
          '--project=my-gcp-project',
      ])


def create_pipeline_components(
    pipeline_root: str,
    transform_module: str,
    trainer_module: str,
    bigquery_query: str = '',
    csv_input_location: str = '',
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
    example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=bigquery_query)
  else:
    example_gen = tfx.components.CsvExampleGen(input_base=csv_input_location)

  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])
  schema_gen = tfx.components.SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  example_validator = tfx.components.ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  transform = tfx.components.Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=transform_module)
  latest_model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestArtifactStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model)).with_id(
          'Resolver.latest_model_resolver')
  trainer = tfx.components.Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(Executor),
      examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      base_model=latest_model_resolver.outputs['model'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=tfx.proto.TrainArgs(num_steps=10),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      module_file=trainer_module,
  )
  # Get the latest blessed model for model validation.
  model_resolver = tfx.dsl.Resolver(
      strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
      model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
      model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'Resolver.latest_blessed_model_resolver')
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
  evaluator = tfx.components.Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  with conditional.Cond(evaluator.outputs['blessing'].future()
                        [0].custom_property('blessed') == 1):
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=os.path.join(pipeline_root, 'model_serving'))))

  return [
      example_gen, statistics_gen, schema_gen, example_validator, transform,
      latest_model_resolver, trainer, model_resolver, evaluator, pusher
  ]


# TODO(b/158245564): Reevaluate whether to keep this test helper function
def full_taxi_pipeline() -> tfx.dsl.Pipeline:
  """Returns a full taxi pipeline under test."""
  pipeline_components = create_pipeline_components(
      pipeline_root=_TEST_PIPELINE_ROOT,
      transform_module=_TEST_MODULE_FILE_LOCATION,
      trainer_module=_TEST_MODULE_FILE_LOCATION,
      csv_input_location=_TEST_INPUT_DATA)

  return tfx.dsl.Pipeline(
      pipeline_name=_TEST_FULL_PIPELINE_NAME,
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=pipeline_components)


class TransformerSpec(component_spec.ComponentSpec):
  """ComponentSpec for a dummy container component."""
  INPUTS = {
      'input1':
          component_spec.ChannelParameter(
              type=tfx.types.standard_artifacts.Model),
  }
  OUTPUTS = {
      'output1':
          component_spec.ChannelParameter(
              type=tfx.types.standard_artifacts.Model),
  }
  PARAMETERS = {
      'param1': component_spec.ExecutionParameter(type=str),
  }


class ProducerSpec(component_spec.ComponentSpec):
  INPUTS = {}
  OUTPUTS = {
      'output1':
          component_spec.ChannelParameter(
              type=tfx.types.standard_artifacts.Model),
  }
  PARAMETERS = {
      'param1': component_spec.ExecutionParameter(type=str),
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
    super().__init__(spec=spec)
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
    super().__init__(spec=spec)
    if instance_name:
      self._id = '{}.{}'.format(self.__class__.__name__, instance_name)
    else:
      self._id = self.__class__.__name__


dummy_transformer_component = tfx.dsl.experimental.create_container_component(
    name='DummyContainerSpecComponent',
    inputs={
        'input1': tfx.types.standard_artifacts.Model,
    },
    outputs={
        'output1': tfx.types.standard_artifacts.Model,
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


dummy_exit_handler = tfx.dsl.experimental.create_container_component(
    name='ExitHandlerComponent',
    parameters={
        'param1': str,
    },
    image='dummy/producer',
    command=[
        'producer',
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

dummy_producer_component = tfx.dsl.experimental.create_container_component(
    name='DummyProducerComponent',
    outputs={
        'output1': tfx.types.standard_artifacts.Model,
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

dummy_producer_component_2 = tfx.dsl.experimental.create_container_component(
    name='DummyProducerComponent2',
    outputs={
        'output1': tfx.types.standard_artifacts.Model,
    },
    parameters={
        'param1': str,
    },
    image='dummy/producer',
    command=[
        'producer',
    ],
)

dummy_consumer_component = tfx.dsl.experimental.create_container_component(
    name='DummyConsumerComponent',
    inputs={
        'input1': tfx.types.standard_artifacts.Model,
    },
    outputs={
        'output1': tfx.types.standard_artifacts.Model,
    },
    parameters={
        'param1': int,
    },
    image='dummy/consumer',
    command=[
        'consumer',
    ],
)


def pipeline_with_one_container_spec_component() -> tfx.dsl.Pipeline:
  """Pipeline with container."""

  importer_task = tfx.dsl.Importer(
      source_uri='some-uri',
      artifact_type=tfx.types.standard_artifacts.Model,
  ).with_id('my_importer')

  container_task = DummyContainerSpecComponent(
      input1=importer_task.outputs['result'],
      output1=channel_utils.as_channel([tfx.types.standard_artifacts.Model()]),
      param1='value1',
  )

  return tfx.dsl.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[importer_task, container_task],
  )


def pipeline_with_two_container_spec_components() -> tfx.dsl.Pipeline:
  """Pipeline with container."""

  container1_task = DummyProducerComponent(
      output1=channel_utils.as_channel([tfx.types.standard_artifacts.Model()]),
      param1='value1',
  )

  container2_task = DummyContainerSpecComponent(
      input1=container1_task.outputs['output1'],
      output1=channel_utils.as_channel([tfx.types.standard_artifacts.Model()]),
      param1='value2',
  )

  return tfx.dsl.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[container1_task, container2_task],
  )


def pipeline_with_two_container_spec_components_2() -> tfx.dsl.Pipeline:
  """Pipeline with container."""

  container1_task = dummy_producer_component(
      output1=channel_utils.as_channel([tfx.types.standard_artifacts.Model()]),
      param1='value1',
  )

  container2_task = dummy_transformer_component(
      input1=container1_task.outputs['output1'],
      output1=channel_utils.as_channel([tfx.types.standard_artifacts.Model()]),
      param1='value2',
  )

  return tfx.dsl.Pipeline(
      pipeline_name='pipeline-with-container',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[container1_task, container2_task],
  )


def get_proto_from_test_data(filename: str,
                             pb_message: message.Message) -> message.Message:
  """Helper function that gets proto from testdata."""
  filepath = os.path.join(os.path.dirname(__file__), 'testdata', filename)
  return tfx.utils.parse_pbtxt_file(filepath, pb_message)


def get_text_from_test_data(filename: str) -> str:
  """Helper function that gets raw string from testdata."""
  filepath = os.path.join(os.path.dirname(__file__), 'testdata', filename)
  return tfx.dsl.io.fileio.open(filepath, 'rb').read().decode('utf-8')


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
    super().__init__(
        _ProducerComponentSpec(
            examples=examples_channel, external_data=external_data_channel))


class ConsumerComponent(base_component.BaseComponent):
  """Test component used in step 2 of a 2-step pipeline testing AI Platform simple artifact types."""
  SPEC_CLASS = _ConsumerComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
      executor_class=base_executor.EmptyExecutor)

  def __init__(self, examples: tfx.dsl.Channel, external_data: tfx.dsl.Channel):
    stats_output_channel = channel_utils.as_channel(
        [simple_artifacts.Statistics()])
    metrics_output_channel = channel_utils.as_channel(
        [simple_artifacts.Metrics()])
    super().__init__(
        _ConsumerComponentSpec(
            examples=examples,
            external_data=external_data,
            stats=stats_output_channel,
            metrics=metrics_output_channel))


def two_step_kubeflow_artifacts_pipeline() -> tfx.dsl.Pipeline:
  """Builds 2-Step pipeline to test AI Platform simple artifact types."""
  step1 = ProducerComponent()
  step2 = ConsumerComponent(
      examples=step1.outputs['examples'],
      external_data=step1.outputs['external_data'])
  return tfx.dsl.Pipeline(
      pipeline_name='two-step-kubeflow-artifacts-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[step1, step2],
      beam_pipeline_args=[
          '--project=my-gcp-project',
      ])


def two_step_pipeline_with_task_only_dependency() -> tfx.dsl.Pipeline:
  """Returns a simple 2-step pipeline with task only dependency between them."""

  step_1 = tfx.dsl.experimental.create_container_component(
      name='Step 1',
      inputs={},
      outputs={},
      parameters={},
      image='step-1-image',
      command=['run', 'step-1'])()

  step_2 = tfx.dsl.experimental.create_container_component(
      name='Step 2',
      inputs={},
      outputs={},
      parameters={},
      image='step-2-image',
      command=['run', 'step-2'])()
  step_2.add_upstream_node(step_1)

  return tfx.dsl.Pipeline(
      pipeline_name='two-step-task-only-dependency-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[step_1, step_2],
  )


primitive_producer_component = tfx.dsl.experimental.create_container_component(
    name='ProducePrimitives',
    outputs={
        'output_string': tfx.types.standard_artifacts.String,
        'output_int': tfx.types.standard_artifacts.Integer,
        'output_float': tfx.types.standard_artifacts.Float,
    },
    image='busybox',
    command=[
        'produce',
        placeholders.OutputUriPlaceholder('output_string'),
        placeholders.OutputUriPlaceholder('output_int'),
        placeholders.OutputUriPlaceholder('output_float'),
    ],
)

primitive_consumer_component = tfx.dsl.experimental.create_container_component(
    name='ConsumeByValue',
    inputs={
        'input_string': tfx.types.standard_artifacts.String,
        'input_int': tfx.types.standard_artifacts.Integer,
        'input_float': tfx.types.standard_artifacts.Float,
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


def consume_primitive_artifacts_by_value_pipeline() -> tfx.dsl.Pipeline:
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

  return tfx.dsl.Pipeline(
      pipeline_name='consume-primitive-artifacts-by-value-pipeline',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[producer_task, consumer_task],
  )


def pipeline_with_runtime_parameter() -> tfx.dsl.Pipeline:
  """Pipeline which contains a runtime parameter."""

  producer_task = primitive_producer_component()

  consumer_task = primitive_consumer_component(
      input_string=producer_task.outputs['output_string'],
      input_int=producer_task.outputs['output_int'],
      input_float=producer_task.outputs['output_float'],
      param_string=tfx.dsl.experimental.RuntimeParameter(
          ptype=str, name='string_param', default='string value'),
      param_int=42,
      param_float=3.14,
  )

  return tfx.dsl.Pipeline(
      pipeline_name='pipeline-with-runtime-parameter',
      pipeline_root=_TEST_PIPELINE_ROOT,
      components=[producer_task, consumer_task],
  )
