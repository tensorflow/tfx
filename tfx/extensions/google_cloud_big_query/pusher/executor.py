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
"""Custom executor to push TFX model to Big Query."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, List, Text

from absl import logging
from google.cloud import bigquery
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx.utils import telemetry_utils

_POLLING_INTERVAL_IN_SECONDS = 30

_GCS_PREFIX = 'gs://'

# Keys to the items in custom_config passed as a part of exec_properties.
SERVING_ARGS_KEY = 'bigquery_serving_args'

# BigQueryML serving argument keys
_PROJECT_ID_KEY = 'project_id'
_BQ_DATASET_ID_KEY = 'bq_dataset_id'
_MODEL_NAME_KEY = 'model_name'

# Keys for custom_config.
_CUSTOM_CONFIG_KEY = 'custom_config'

# Model name should be enclosed within backticks.
# model_path should ends with asterisk glob (/*).
_BQML_CREATE_OR_REPLACE_MODEL_QUERY_TEMPLATE = """
CREATE OR REPLACE MODEL `{model_uri}`
OPTIONS (model_type='tensorflow',
         model_path='{model_path}/*')
"""


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to BigQuery ML for serving."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]):
    """Overrides the tfx_pusher_executor.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from evaluator.
      output_dict: Output dict from key to a list of artifacts, including:
        - model_push: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: Mostly a passthrough input dict for
        tfx.components.Pusher.executor.  custom_config.bigquery_serving_args is
        consumed by this class.  For the full set of parameters supported by
        Big Query ML, refer to https://cloud.google.com/bigquery-ml/

    Returns:
      None
    Raises:
      ValueError:
        If bigquery_serving_args is not in exec_properties.custom_config.
        If pipeline_root is not 'gs://...'
      RuntimeError: if the Big Query job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY])
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return

    model_export = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.MODEL_KEY])
    model_export_uri = model_export.uri

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')

    bigquery_serving_args = custom_config.get(SERVING_ARGS_KEY)
    # if configuration is missing error out
    if bigquery_serving_args is None:
      raise ValueError('Big Query ML configuration was not provided')

    bq_model_uri = '.'.join([
        bigquery_serving_args[_PROJECT_ID_KEY],
        bigquery_serving_args[_BQ_DATASET_ID_KEY],
        bigquery_serving_args[_MODEL_NAME_KEY],
    ])

    # Deploy the model.
    io_utils.copy_dir(
        src=path_utils.serving_model_path(
            model_export_uri, path_utils.is_old_model_artifact(model_export)),
        dst=model_push.uri)
    model_path = model_push.uri
    if not model_path.startswith(_GCS_PREFIX):
      raise ValueError('pipeline_root must be gs:// for BigQuery ML Pusher.')

    logging.info('Deploying the model to BigQuery ML for serving: %s from %s',
                 bigquery_serving_args, model_path)

    query = _BQML_CREATE_OR_REPLACE_MODEL_QUERY_TEMPLATE.format(
        model_uri=bq_model_uri, model_path=model_path)

    # TODO(zhitaoli): Refactor the executor_class_path creation into a common
    # utility function.
    executor_class_path = '%s.%s' % (self.__class__.__module__,
                                     self.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      default_query_job_config = bigquery.job.QueryJobConfig(
          labels=telemetry_utils.get_labels_dict())
    # TODO(b/181368842) Add integration test for BQML Pusher + Managed Pipeline
    client = bigquery.Client(
        default_query_job_config=default_query_job_config,
        project=bigquery_serving_args[_PROJECT_ID_KEY])

    try:
      query_job = client.query(query)
      query_job.result()  # Waits for the query to finish
    except Exception as e:
      raise RuntimeError('BigQuery ML Push failed: {}'.format(e))

    logging.info('Successfully deployed model %s serving from %s', bq_model_uri,
                 model_path)

    # Setting the push_destination to bigquery uri
    self._MarkPushed(model_push, pushed_destination=bq_model_uri)
