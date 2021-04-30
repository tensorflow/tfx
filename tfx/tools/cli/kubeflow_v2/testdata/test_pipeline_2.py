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
"""Chicago taxi example using TFX on Kubeflow pipelines V2 runner."""

import os
from typing import Text

import absl
from absl import app

from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

_pipeline_name = 'chicago-taxi-kubeflow'
_taxi_root = os.path.join(os.environ['HOME'], 'taxi')
_data_root = os.path.join(_taxi_root, 'data', 'simple')
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)


def _create_pipeline(pipeline_name: Text, pipeline_root: Text,
                     data_root: Text) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=data_root)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[example_gen],
      enable_cache=True,
      additional_pipeline_args={},
  )


def main(_):
  absl.logging.set_verbosity(absl.logging.INFO)

  dsl_pipeline = _create_pipeline(
      pipeline_name=_pipeline_name,
      pipeline_root=_pipeline_root,
      data_root=_data_root)

  runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig()

  runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(config=runner_config)
  runner.run(pipeline=dsl_pipeline)

if __name__ == '__main__':
  app.run(main)
