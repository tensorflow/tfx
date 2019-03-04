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
"""Definition of Airflow TFX runner."""
from tfx.orchestration import tfx_runner
from tfx.orchestration.airflow import airflow_component
from tfx.orchestration.airflow import airflow_pipeline


class AirflowDAGRunner(tfx_runner.TfxRunner):
  """Tfx runner on Airflow."""

  def __init__(self, config=None):
    super(AirflowDAGRunner, self).__init__()
    self._config = config or {}

  def _prepare_input_dict(self, input_dict):
    return dict((k, v.get()) for k, v in input_dict.items())

  def _prepare_output_dict(self, outputs):
    return dict((k, v.get()) for k, v in outputs.get_all().items())

  def run(self, pipeline):
    """Deploys given logical pipeline on Airflow.

    Args:
      pipeline: Logical pipeline containing pipeline args and components.

    Returns:
      An Airflow DAG.
    """

    # Merge airflow-specific configs with pipeline args
    self._config.update(pipeline.pipeline_args)
    airflow_dag = airflow_pipeline.AirflowPipeline(**self._config)

    # For every components in logical pipeline, add in real component.
    for component in pipeline.components:
      airflow_component.Component(
          airflow_dag,
          component_name=component.component_name,
          unique_name=component.unique_name,
          driver=component.driver,
          executor=component.executor,
          input_dict=self._prepare_input_dict(component.input_dict),
          output_dict=self._prepare_output_dict(component.outputs),
          exec_properties=component.exec_properties)

    return airflow_dag
