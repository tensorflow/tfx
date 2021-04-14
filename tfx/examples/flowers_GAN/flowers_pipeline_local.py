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
"""Flower example using TFX."""

import os
import sys
from typing import List, Optional, Text

import absl
from absl import flags

from tfx.components import ImportExampleGen
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components import Tuner
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

flags.DEFINE_enum(
    'model_framework', 'keras', ['keras', 'flax_experimental'],
    'The modeling framework.')

# This example assumes that flowers data is stored in ./data and the
# utility function is in the current folder. Feel free to customize as needed.
_flowers_root = '.'
_data_root = os.path.join(_flowers_root, 'data')

# Directory and data locations.  This example assumes all of the
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]


def _create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_root: Text,
        module_file: Text,
        serving_model_dir: Text,
        metadata_path: Text,
        train_steps: int,
        eval_steps: int,
        beam_pipeline_args: List[Text],
        enable_cache: Optional[bool] = True
) -> pipeline.Pipeline:
    """Implements the penguin pipeline with TFX.

    Args:
      pipeline_name: name of the TFX pipeline being created.
      pipeline_root: root directory of the pipeline.
      data_root: directory containing the penguin data.
      module_file: path to files used in Trainer and Transform components.
      serving_model_dir: filepath to write pipeline SavedModel to.
      metadata_path: path to local pipeline ML Metadata store.
      beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
        refer to https://beam.apache.org/documentation/runners/direct/.
      enable_cache: Optional boolean

    Returns:
      A TFX pipeline object.
    """
    # Brings data into the pipeline and splits the data into training and eval splits
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
        ]))

    example_gen = ImportExampleGen(input_base=data_root,
                                   output_config=output_config)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                           infer_feature_shape=True)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        examples=example_gen.outputs.examples,
        schema=schema_gen.outputs.schema,
        module_file=module_file)

    # Tunes the hyperparameters for model training based on user-provided Python
    # function. Note that once the hyperparameters are tuned, you can drop the
    # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
    # TODO: add tuner support.
    # if enable_tuning:
    #     tuner = Tuner(
    #         module_file=module_file,
    #         examples=transform.outputs['transformed_examples'],
    #         transform_graph=transform.outputs['transform_graph'],
    #         train_args=trainer_pb2.TrainArgs(num_steps=20),
    #         eval_args=trainer_pb2.EvalArgs(num_steps=5))

    # TODO: add tuner support.
    # Uses user-provided Python function that trains a model.
    trainer = Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps))

    # Always push the model to a file destination.
    # TODO: Add model_blessing with Evaluator to check whether the model passed
    #  the validation steps before pushing the model.
    pusher = Pusher(
        model=trainer.outputs['model'],
        # model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        pusher,
    ]

    # TODO: enable tuning.
    # if enable_tuning:
    #     components.append(tuner)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python penguin_pipeline_local.py [--model_framework=flax_experimental]
if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)
    absl.flags.FLAGS(sys.argv)
    _pipeline_name = f'flowers_local_{flags.FLAGS.model_framework}'

    # Python module file to inject customized logic into the TFX components. The
    # Transform, Trainer and Tuner all require user-defined functions to run
    # successfully.
    _module_file_name = f'flowers_utils_{flags.FLAGS.model_framework}.py'
    _module_file = os.path.join(_flowers_root, _module_file_name)
    # Path which can be listened to by the model server.  Pusher will output the
    # trained model here.
    _serving_model_dir = os.path.join(_flowers_root, 'serving_model',
                                      _pipeline_name)
    # Pipeline root for artifacts.
    _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
    # Sqlite ML-metadata db path.
    _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                  'metadata.db')
    LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=_data_root,
            module_file=_module_file,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
            # TODO(b/180723394): support tuning for Flax.
            # enable_tuning=(flags.FLAGS.model_framework == 'keras'),
            train_steps=30,
            eval_steps=10,
            beam_pipeline_args=_beam_pipeline_args))
