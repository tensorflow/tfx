# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Object detection TFX pipeline."""
import os
from typing import List

from tfx import v1 as tfx

_pipeline_name = 'YOLO_object_detection'
_data_root = os.path.join(os.environ['HOME'], 'object_detection', 'data')
# Directory and data locations.  This example assumes all the example code and
# metadata library is relative to $HOME, but you can store these files anywhere
# on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
# Sqlite ML-metadata db path.
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')
_module_file = os.path.join(
        os.path.dirname(__file__), 'object_detection_utils.py')
# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, metadata_path: str,
                     beam_pipeline_args: List[str]) -> tfx.dsl.Pipeline:
  """Implements the object detection pipeline with TFX.

  Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline.
    data_root: directory containing the training data.
    module_file: module file containing training util functions.
    metadata_path: path to local pipeline ML Metadata store.
    beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
      refer to https://beam.apache.org/documentation/runners/direct/.

  Returns:
    A TFX pipeline object.
  """
  input_config = tfx.proto.Input(
      splits=[tfx.proto.Input.Split(name='all', pattern='*')])
  output_config = tfx.proto.Output(
      split_config=tfx.proto.SplitConfig(splits=[
          tfx.proto.SplitConfig.Split(name='train', hash_buckets=8),
          tfx.proto.SplitConfig.Split(name='eval', hash_buckets=2)
      ]))

  example_gen = tfx.components.ImportExampleGen(
      input_base=data_root,
      input_config=input_config,
      output_config=output_config
      )

  statistics_gen = tfx.components.StatisticsGen(
      examples=example_gen.outputs['examples'])

  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      hyperparameters=None,
      train_args=tfx.proto.TrainArgs(num_steps=2),
      eval_args=tfx.proto.EvalArgs(num_steps=1))

  components_list = [
      example_gen,
      statistics_gen,
      trainer
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components_list,
      metadata_connection_config=tfx.orchestration.metadata
      .sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_pipeline_args
      )

if __name__ == '__main__':

  tfx.orchestration.LocalDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          pipeline_root=_pipeline_root,
          data_root=_data_root,
          metadata_path=_metadata_path,
          module_file=_module_file,
          beam_pipeline_args=_beam_pipeline_args,
          ))

