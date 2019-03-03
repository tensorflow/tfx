# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chicago Taxi example using TFX DSL on Kubeflow."""
import os
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.kubeflow.runner import KubeflowRunner as TfxRunner
from tfx.orchestration.pipeline import PipelineDecorator
from tfx.utils.dsl_utils import csv_input

# Cloud storage.
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'
# Data location
_base_dir = os.path.join(_input_bucket, 'data/taxi_data')
# Helper functions for the taxi pipleine: estimator and preprocessing_fn
_taxi_pipeline_utils = os.path.join(_input_bucket, 'taxi_utils.py')
# Path which can be listened by model server. Pusher will output model here.
_serving_model_dir = os.path.join(_input_bucket, 'serving_model/taxi_bigquery')
# Root for all pipeline output.
_pipeline_root = os.path.join(_output_bucket, 'output')


@PipelineDecorator(
    pipeline_name='chicago_taxi_pipeline_kubeflow',
    log_root='/var/tmp/tfx/logs',
    pipeline_root=_pipeline_root)
def create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_input(os.path.join(_base_dir, 'no_split/span_1'))

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=_taxi_pipeline_utils)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=_taxi_pipeline_utils,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_steps=10000,
      eval_steps=5000,
      warm_starting=True)

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output)

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      serving_model_dir=_serving_model_dir)

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


pipeline = TfxRunner().run(create_pipeline())
