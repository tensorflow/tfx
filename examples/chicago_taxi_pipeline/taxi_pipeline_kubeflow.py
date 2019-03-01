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
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.runtimes.kubeflow.runner import KubeflowRunner as TfxRunner
from tfx.runtimes.pipeline import PipelineDecorator
from tfx.utils.dsl_utils import csv_inputs

# Cloud storage.
input_bucket = 'gs://my-bucket'
output_bucket = 'gs://my-bucket'
# Data location
base_dir = os.path.join(input_bucket, 'data/taxi_data')
# Helper functions for the taxi pipleine: estimator and preprocessing_fn
taxi_pipeline_utils = os.path.join(input_bucket, 'taxi_utils.py')
# Path which can be listened by model server. Pusher will output model here.
serving_model_dir = os.path.join(input_bucket, 'serving_model/taxi_bigquery')
# Root for all pipeline output.
pipeline_root = os.path.join(output_bucket, 'output')


@PipelineDecorator(
    pipeline_name='chicago_taxi_pipeline_kubeflow',
    log_root='/var/tmp/tfx/logs',
    pipeline_root=pipeline_root)
def create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_inputs(os.path.join(base_dir, 'no_split/span_1'))

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_data=examples)

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
      module_file=taxi_pipeline_utils)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=taxi_pipeline_utils,
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
      serving_model_dir=serving_model_dir)

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


pipeline = TfxRunner().run(create_pipeline())
