# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Chicago taxi example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Dict, List, Text

import absl
import tensorflow_model_analysis as tfma
import tensorflow as tf

from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.types import standard_artifacts
from tfx.types import artifact_utils
from tfx.types.artifact import Artifact
from tfx.utils.dsl_utils import external_input
from tfx.experimental.mock_units.mock_factory import FakeComponentExecutorFactory


class Test(tf.test.TestCase):

  def testMockExecutor(self):
    def _create_pipeline(pipeline_name: Text, pipeline_root: Text,
                         data_root: Text, module_file: Text,
                         serving_model_dir: Text,
                         metadata_path: Text,
                         direct_num_workers: int) -> pipeline.Pipeline:
      """Implements the chicago taxi pipeline with TFX."""
      examples = external_input(data_root)

      # Brings data into the pipeline or otherwise joins/converts training data.
      example_gen = CsvExampleGen(input=examples)

      # Computes statistics over data for visualization and example validation.
      statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

      # Generates schema based on statistics files.
      schema_gen = SchemaGen(
          statistics=statistics_gen.outputs['statistics'],
          infer_feature_shape=False)

      # Performs anomaly detection based on statistics and data schema.
      example_validator = ExampleValidator(
          statistics=statistics_gen.outputs['statistics'],
          schema=schema_gen.outputs['schema'])

      # Performs transformations and feature engineering in training and serving.
      transform = Transform(
          examples=example_gen.outputs['examples'],
          schema=schema_gen.outputs['schema'],
          module_file=module_file)

      # Uses user-provided Python function that implements a model using TF-Learn.
      trainer = Trainer(
          module_file=module_file,
          transformed_examples=transform.outputs['transformed_examples'],
          schema=schema_gen.outputs['schema'],
          transform_graph=transform.outputs['transform_graph'],
          train_args=trainer_pb2.TrainArgs(num_steps=10000),
          eval_args=trainer_pb2.EvalArgs(num_steps=5000))

      # Get the latest blessed model for model validation.
      model_resolver = ResolverNode(
          instance_name='latest_blessed_model_resolver',
          resolver_class=LatestBlessedModelResolver,
          model=Channel(type=Model),
          model_blessing=Channel(type=ModelBlessing))

      # Uses TFMA to compute a evaluation statistics over features of a model and
      # perform quality validation of a candidate model (compared to a baseline).
      eval_config = tfma.EvalConfig(
          model_specs=[tfma.ModelSpec(signature_name='eval')],
          slicing_specs=[
              tfma.SlicingSpec(),
              tfma.SlicingSpec(feature_keys=['trip_start_hour'])
          ],
          metrics_specs=[
              tfma.MetricsSpec(
                  thresholds={
                      'accuracy':
                          tfma.config.MetricThreshold(
                              value_threshold=tfma.GenericValueThreshold(
                                  lower_bound={'value': 0.6}),
                              change_threshold=tfma.GenericChangeThreshold(
                                  direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                  absolute={'value': -1e-10}))
                  })
          ])
      evaluator = Evaluator(
          examples=example_gen.outputs['examples'],
          model=trainer.outputs['model'],
          baseline_model=model_resolver.outputs['model'],
          # Change threshold will be ignored if there is no baseline (first run).
          eval_config=eval_config)

      # Checks whether the model passed the validation steps and pushes the model
      # to a file destination if check passed.
      pusher = Pusher(
          model=trainer.outputs['model'],
          model_blessing=evaluator.outputs['blessing'],
          push_destination=pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)))


      return pipeline.Pipeline(
          pipeline_name=pipeline_name,
          pipeline_root=pipeline_root,
          components=[
              example_gen,
              statistics_gen,
              schema_gen,
              example_validator,
              transform,
              trainer,
              model_resolver,
              evaluator,
              pusher,
          ],
          # enable_cache=True,
          metadata_connection_config=metadata.sqlite_metadata_connection_config(
              metadata_path),
          # TODO(b/142684737): The multi-processing API might change.
          beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers])
    def _compare_artifacts(expected_artifacts: Dict[Text, List[Artifact]],
                           artifacts: Dict[Text, List[Artifact]]):
      for component_key, expected_artifact_list in expected_artifacts.items():
        self.assertIn(component_key, artifacts, 
                      msg="{} missing".format(component_key))

        output_artifacts_list = artifacts[component_key].get()
        for output_artifact in output_artifacts_list:
          for expected_artifact in expected_artifact_list:
            output_artifact_name = output_artifact._artifact_type.name
            expected_artifact_name = expected_artifact._artifact_type.name
            if output_artifact_name == expected_artifact_name:
              self.assertProtoEquals(
                  output_artifact.artifact_type, expected_artifact.artifact_type)
              self.assertProtoEquals(
                  output_artifact.mlmd_artifact, expected_artifact.mlmd_artifact)
              self.assertProtoEquals(output_artifact.uri, expected_artifact.uri)
              break
          else:
            self.fail("Artifacts don't match")

    pipeline_name = 'chicago_taxi_beam'

    # This example assumes that the taxi data is stored in ~/taxi/data and the
    # taxi utility function is in ~/taxi.  Feel free to customize this as needed.
    # _taxi_root = os.path.join(os.environ['HOME'], 'taxi')
    taxi_root = '/usr/local/google/home/sujip/tfx/tfx/examples/chicago_taxi_pipeline'
    data_root = os.path.join(taxi_root, 'data', 'simple')
    # Python module file to inject customized logic into the TFX components. The
    # Transform and Trainer both require user-defined functions to run successfully.
    module_file = os.path.join(taxi_root, 'taxi_utils.py')
    # Path which can be listened to by the model server.  Pusher will output the
    # trained model here.
    serving_model_dir = os.path.join(taxi_root, 'serving_model',
                                     pipeline_name)

    # Directory and data locations.  This example assumes all of the chicago taxi
    # example code and metadata library is relative to $HOME, but you can store
    # these files anywhere on your local filesystem.
    tfx_root = os.path.join(os.environ['HOME'], 'tfx')
    pipeline_root = os.path.join(tfx_root, 'pipelines', pipeline_name)
    # Sqlite ML-metadata db path.
    metadata_path = os.path.join(tfx_root, 'metadata', pipeline_name,
                                 'metadata.db')

    # Creating artifacts
    external_artifact = standard_artifacts.ExternalArtifact()
    external_artifact.uri = ''

    examples = standard_artifacts.Examples()
    examples.uri = ''
    examples.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    # example_statistics = standard_artifacts.ExampleStatistics()
    # example_statistics.uri = ''
    # statistics.split_names = artifact_utils.encode_split_names(['train', 'eval'])

    # schema = standard_artifacts.Schema()
    # schema.uri = ''

    # anomalies=standard_artifacts.ExampleAnomalies()
    # anomalies.uri = ''

    # transform_graph = standard_artifacts.TransformGraph()
    # transform_graph.uri = ''

    # transformed_examples = standard_artifacts.Examples()
    # transformed_examples.uri = ''

    # model = standard_artifacts.Model()
    # model.uri = ''

    # baseline_model = standard_artifacts.Model()

    # evaluation = standard_artifacts.ModelEvaluation()
    # evaluation.uri =''

    # model_blessing = standard_artifacts.ModelBlessing()
    # model_blessing.uri = ''

    # pushed_model = standard_artifacts.PushedModel()
    # pushed_model.uri = ''

    mock_pipeline = _create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_root=data_root,
        module_file=module_file,
        serving_model_dir=serving_model_dir,
        metadata_path=metadata_path,
        direct_num_workers=0) # pipeline dsl

    mock_pipeline.set_executor('CsvExampleGen', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('StatisticsGen', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('SchemaGen', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('ExampleValidator', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('Transform', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('Trainer', FakeComponentExecutorFactory)
    # mock_pipeline.set_executor('ResolverNode.latest_blessed_model_resolver',...)
    mock_pipeline.set_executor('Evaluator', FakeComponentExecutorFactory)
    mock_pipeline.set_executor('Pusher', FakeComponentExecutorFactory)

    BeamDagRunner().run(mock_pipeline)

    csvgen_input = mock_pipeline.get_artifacts('CsvExampleGen')['input_dict']
    csvgen_output = mock_pipeline.get_artifacts('CsvExampleGen')['output_dict']
    _compare_artifacts({'input': [external_artifact]}, csvgen_input)
    _compare_artifacts({'examples': [examples]}, csvgen_output)


if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)
  tf.test.main()
