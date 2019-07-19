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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from airflow import models

from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration.airflow import airflow_component


class AirflowDagRunner(tfx_runner.TfxRunner):
  """Tfx runner on Airflow."""

  def __init__(self, config=None):
    super(AirflowDagRunner, self).__init__()
    self._config = config or {}

  def run(self, tfx_pipeline: pipeline.Pipeline):
    """Deploys given logical pipeline on Airflow.

    Args:
      tfx_pipeline: Logical pipeline containing pipeline args and components.

    Returns:
      An Airflow DAG.
    """

    # Merge airflow-specific configs with pipeline args
    airflow_dag = models.DAG(
        dag_id=tfx_pipeline.pipeline_info.pipeline_name, **self._config)
    if 'tmp_dir' not in tfx_pipeline.additional_pipeline_args:
      tmp_dir = os.path.join(tfx_pipeline.pipeline_info.pipeline_root, '.temp',
                             '')
      tfx_pipeline.additional_pipeline_args['tmp_dir'] = tmp_dir

    component_impl_map = {}
    for tfx_component in tfx_pipeline.components:
      current_airflow_component = airflow_component.AirflowComponent(
          airflow_dag,
          component=tfx_component,
          pipeline_info=tfx_pipeline.pipeline_info,
          enable_cache=tfx_pipeline.enable_cache,
          metadata_connection_config=tfx_pipeline.metadata_connection_config,
          additional_pipeline_args=tfx_pipeline.additional_pipeline_args)
      component_impl_map[tfx_component] = current_airflow_component
      for upstream_node in tfx_component.upstream_nodes:
        assert upstream_node in component_impl_map, ('Components is not in '
                                                     'topological order')
        current_airflow_component.set_upstream(
            component_impl_map[upstream_node])

    return airflow_dag
