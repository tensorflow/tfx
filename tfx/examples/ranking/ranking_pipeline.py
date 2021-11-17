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
"""tf.ranking example pipeline."""

import os
from typing import List

import absl
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ImportExampleGen
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.experimental.data_view import binder_component
from tfx.components.experimental.data_view import provider_component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import transform_pb2

_pipeline_name = 'tf_ranking_antique'

# This example assumes that the training data is stored in
# ~/tf_ranking_antique/data
# and the module file is in ~/tf_ranking_antique.  Feel free to customize this
# as needed.
_ranking_root = os.path.join(os.environ['HOME'], 'tf_ranking_antique')
_data_root = os.path.join(_ranking_root, 'data')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_module_file = os.path.join(_ranking_root, 'ranking_utils.py')
# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(
    _ranking_root, 'serving_model', _pipeline_name)

# Directory and data locations.  This example assumes all the example code and
# metadata library is relative to $HOME, but you can store these files anywhere
# on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str, beam_pipeline_args: List[str]):
  """Creates pipeline."""
  pipeline_root = os.path.join(pipeline_root, 'pipelines', pipeline_name)

  example_gen = ImportExampleGen(
      input_base=data_root,
      # IMPORTANT: must set FORMAT_PROTO
      payload_format=example_gen_pb2.FORMAT_PROTO)

  data_view_provider = provider_component.TfGraphDataViewProvider(
      module_file=module_file,
      create_decoder_func='make_decoder')

  data_view_binder = binder_component.DataViewBinder(
      example_gen.outputs['examples'],
      data_view_provider.outputs['data_view'])

  statistics_gen = StatisticsGen(
      examples=data_view_binder.outputs['output_examples'])

  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

  transform = Transform(
      examples=data_view_binder.outputs['output_examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file,
      # important: must disable Transform materialization and ensure the
      # transform field of the splits config is empty.
      splits_config=transform_pb2.SplitsConfig(analyze=['train']),
      materialize=False)

  trainer = Trainer(
      examples=data_view_binder.outputs['output_examples'],
      transform_graph=transform.outputs['transform_graph'],
      module_file=module_file,
      train_args=trainer_pb2.TrainArgs(num_steps=1000),
      schema=schema_gen.outputs['schema'],
      eval_args=trainer_pb2.EvalArgs(num_steps=10))

  eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='',
              label_key='relevance',
              padding_options=tfma.PaddingOptions(
                  label_float_padding=-1.0, prediction_float_padding=-1.0))
      ],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['query_tokens']),
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              per_slice_thresholds={
                  'metric/ndcg_10':
                      tfma.PerSliceMetricThresholds(thresholds=[
                          tfma.PerSliceMetricThreshold(
                              # The overall slice.
                              slicing_specs=[tfma.SlicingSpec()],
                              threshold=tfma.MetricThreshold(
                                  value_threshold=tfma.GenericValueThreshold(
                                      lower_bound={'value': 0.6})))
                      ])
              })
      ])

  evaluator = Evaluator(
      examples=data_view_binder.outputs['output_examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config,
      schema=schema_gen.outputs['schema'])

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
          example_gen, data_view_provider, data_view_binder,
          statistics_gen,
          schema_gen,
          transform,
          trainer,
          evaluator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $ python ranking_pipeline.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          module_file=_module_file,
          metadata_path=_metadata_path,
          serving_model_dir=_serving_model_dir,
          beam_pipeline_args=_beam_pipeline_args))
