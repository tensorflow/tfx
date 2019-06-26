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
"""Definition of Airflow TFX adapter."""
import json

from tfx.components.base import base_executor
from tfx.orchestration import metadata
from tfx.utils import logging_utils
from tfx.utils.types import jsonify_tfx_type_dict
from tfx.utils.types import parse_tfx_type_dict
from tfx.utils.types import TfxArtifact


# TODO(b/126566908): More documentation for Airflow modules.
class AirflowAdapter(object):
  """Execute executor based on decision from Metadata."""

  # TODO(khaas): pytypes here
  def __init__(self, component_name, input_dict, output_dict, exec_properties,
               driver_args, driver_class, executor_class,
               additional_pipeline_args, metadata_connection_config,
               logger_config):
    """Constructs an AirflowAdaptor.

    Args:
      component_name: Name of the component in the dag.
      input_dict: a dict from key name to a list of TfxArtifact artifacts.
      output_dict: a dict from key name to a list of TfxArtifact artifacts.
      exec_properties: a dict of execution properties.
      driver_args: an instance of orchestration.data_types.DriverArgs to
        serve as additional args to driver;
      driver_class: Python class of driver;
      executor_class: Python class of executor;
      additional_pipeline_args: a dict of additional pipeline args. Currently
        supporting following keys: beam_pipeline_args.
      metadata_connection_config: configuration for how to connect to metadata.
      logger_config: dict of logging parameters for configuring the logger.
    """
    self._component_name = component_name
    self._input_dict = dict((k, v) for k, v in input_dict.items() if v)
    self._output_dict = output_dict
    self._exec_properties = exec_properties
    self._driver_args = driver_args
    self._driver_class = driver_class
    self._executor_class = executor_class
    self._logger = logging_utils.get_logger(logger_config)
    # Resolve source from input_dict and output_dict to decouple this earlier.
    self._input_source_dict = self._make_source_dict(self._input_dict)
    self._output_source_dict = self._make_source_dict(self._output_dict)
    self._execution_id = None
    self._additional_pipeline_args = additional_pipeline_args or {}
    self._metadata_connection_config = metadata_connection_config

  def _make_source_dict(self, inout_dict):
    """Capture name to source mapping from input/output dict."""
    result = dict()
    for key, artifact_list in inout_dict.items():
      if not artifact_list or not getattr(artifact_list[0], 'source', None):
        continue
      result[key] = artifact_list[0].source
    return result

  def _update_input_dict_from_xcom(self, task_instance):
    """Determine actual input artifacts used from upstream tasks."""
    for key, input_list in self._input_dict.items():
      source = self._input_source_dict.get(key)
      if not source:
        continue
      xcom_result = task_instance.xcom_pull(
          key=source.key, dag_id=source.component_id)
      if not xcom_result:
        self._logger.warn('Cannot resolve xcom source from %s', source)
        continue
      resolved_list = json.loads(xcom_result)
      for index, resolved_json_dict in enumerate(resolved_list):
        input_list[index] = TfxArtifact.parse_from_json_dict(resolved_json_dict)

  def _publish_execution_to_metadata(self):
    with metadata.Metadata(self._metadata_connection_config) as m:
      return m.publish_execution(self._execution_id, self._input_dict,
                                 self._output_dict)

  def _publish_outputs_to_pipeline(self, task_instance, final_output):
    for key, output_list in final_output.items():
      source = self._output_source_dict[key]
      output_dict_list = [o.json_dict() for o in output_list]
      xcom_value = json.dumps(output_dict_list)
      task_instance.xcom_push(key=source.key, value=xcom_value)

  def check_cache_and_maybe_prepare_execution(self, cached_branch,
                                              uncached_branch, **kwargs):
    """Depending on previous run status, run exec or skip."""

    task_instance = kwargs['ti']
    self._update_input_dict_from_xcom(task_instance)

    with metadata.Metadata(self._metadata_connection_config) as m:
      driver = self._driver_class(metadata_handler=m)
      execution_decision = driver.prepare_execution(self._input_dict,
                                                    self._output_dict,
                                                    self._exec_properties,
                                                    self._driver_args)
      if not execution_decision.execution_id:
        self._logger.info(
            'All artifacts found. Publishing to pipeline and skipping executor.'
        )
        self._publish_outputs_to_pipeline(task_instance,
                                          execution_decision.output_dict)
        return cached_branch

      task_instance.xcom_push(
          key='_exec_inputs',
          value=jsonify_tfx_type_dict(execution_decision.input_dict))
      task_instance.xcom_push(
          key='_exec_outputs',
          value=jsonify_tfx_type_dict(execution_decision.output_dict))
      task_instance.xcom_push(
          key='_exec_properties',
          value=json.dumps(execution_decision.exec_properties))
      task_instance.xcom_push(
          key='_execution_id', value=execution_decision.execution_id)

      self._logger.info('No cached execution found. Starting executor.')
      return uncached_branch

  def python_exec(self, cache_task_name, **kwargs):
    """PythonOperator callable to invoke executor."""
    # This is executed in worker-space not runtime-space (i.e. with distributed
    # workers, this runs on the worker node not the controller node).
    task_instance = kwargs['ti']
    self._refresh_execution_args_from_xcom(task_instance, cache_task_name)
    executor_context = base_executor.BaseExecutor.Context(
        beam_pipeline_args=self._additional_pipeline_args.get(
            'beam_pipeline_args'),
        tmp_dir=self._additional_pipeline_args.get('tmp_dir'),
        unique_id=self._execution_id)

    executor = self._executor_class(executor_context)

    # Run executor
    executor.Do(self._input_dict, self._output_dict, self._exec_properties)
    # Airflow's docker_operator chooses 'return_value' so we try to be
    # consistent in case one day we want to add it back.
    task_instance.xcom_push(
        key='return_value', value=jsonify_tfx_type_dict(self._output_dict))

  def _refresh_execution_args_from_xcom(self, task_instance, pushing_task_name):
    """Refresh inputs, outputs and exec_properties from xcom."""
    inputs_str = task_instance.xcom_pull(
        key='_exec_inputs', task_ids=pushing_task_name)
    self._input_dict = parse_tfx_type_dict(inputs_str)

    outputs_str = task_instance.xcom_pull(
        key='_exec_outputs', task_ids=pushing_task_name)
    self._output_dict = parse_tfx_type_dict(outputs_str)

    exec_properties_str = task_instance.xcom_pull(
        key='_exec_properties', task_ids=pushing_task_name)
    self._exec_properties = json.loads(exec_properties_str)

    self._execution_id = task_instance.xcom_pull(
        key='_execution_id', task_ids=pushing_task_name)

  def publish_exec(self, cache_task_name, exec_task_name, **kwargs):
    """Publish artifacts produced in this execution to the pipeline."""
    task_instance = kwargs['ti']
    self._refresh_execution_args_from_xcom(task_instance, cache_task_name)

    # Overwrite outputs from cache with outputs produced by exec operator.
    outputs_str = task_instance.xcom_pull(
        key='return_value', task_ids=exec_task_name)
    self._output_dict = parse_tfx_type_dict(outputs_str)
    final_output = self._publish_execution_to_metadata()
    self._publish_outputs_to_pipeline(task_instance, final_output)
