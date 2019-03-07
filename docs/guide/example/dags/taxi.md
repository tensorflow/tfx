```python
"""Chicago taxi example using TFX DSL."""
# Copyright 2018 Google LLC
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
import datetime
import os
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import ExamplesGen
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.runtimes.tfx_airflow import PipelineDecorator
from tfx.utils.dsl_utils import csv_input

# Directory and data locations
home_dir = os.path.join(os.environ['HOME'], 'airflow/')
output_dir = os.path.join(home_dir, 'data/fred/pipelines/')
base_dir = os.path.join(home_dir, 'data/fred/')

# Modules for trainer and transform
model = os.path.join(home_dir, 'plugins/fred/model.py')
transforms = os.path.join(home_dir, 'plugins/fred/transforms.py')

# Path which can be listened by model server. Pusher will output model here.
serving_model_dir = os.path.join(output_dir, 'fred/serving_model')

# For TFMA evaluation
tfx_example_eval_spec = [
    tfma.SingleSliceSpec(),
    tfma.SingleSliceSpec(columns=['trip_start_hour'])
]

# For ModelValidator
tfx_example_mv_spec = [tfma.SingleSliceSpec()]

@PipelineDecorator(
    pipeline_name='fred_DAG', # Note: Your pipeline_name MUST end in "_DAG"
    schedule_interval=None,
    start_date=datetime.datetime(2018, 1, 1),
    enable_cache=True,
    run_id='fred-run-local',
    log_root='/var/tmp/tfx/logs',
    output_dir=output_dir)
def create_pipeline():
  """Implements the example pipeline with TFX."""
  examples = csv_input(os.path.join(base_dir, 'no_split/span_1'))

  examples_gen = ExamplesGen(input_data=examples)

  statistics_gen = StatisticsGen(input_data=examples_gen.outputs.output)

  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  validate_stats = ExampleValidator(  # pylint: disable=unused-variable
      stats=statistics_gen.outputs.output,
      schema=infer_schema.outputs.output)

  transform = Transform(
      input_data=examples_gen.outputs.output,
      schema=infer_schema.outputs.output,
      module_file=transforms)

  trainer = Trainer(
      module_file=model,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_steps=10000,
      eval_steps=5000,
      warm_starting=True)

  model_analyzer = Evaluator(  # pylint: disable=unused-variable
      examples=examples_gen.outputs.output,
      eval_spec=tfx_example_eval_spec,
      model_exports=trainer.outputs.output)

  model_validator = ModelValidator(
      examples=examples_gen.outputs.output,
      model=trainer.outputs.output,
      eval_spec=tfx_example_mv_spec)

  pusher = Pusher(  # pylint: disable=unused-variable
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      serving_model_dir=serving_model_dir)


pipeline = create_pipeline()  # pylint: disable=assignment-from-no-return
```
