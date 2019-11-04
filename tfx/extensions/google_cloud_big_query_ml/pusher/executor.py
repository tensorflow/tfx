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

import os
import absl
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx import types
from google.cloud import bigquery
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.types import artifact_utils
from tfx.utils import path_utils

_POLLING_INTERVAL_IN_SECONDS = 30


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to BigQuery ML for serving."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]):
    """Overrides the tfx_pusher_executor.

    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - model_export: exported model from trainer.
        - model_blessing: model blessing path from model_validator.
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
      ValueError: if bigquery_serving_args is not in
      exec_properties.custom_config.
      RuntimeError: if the Big Query job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)
    if not self.CheckBlessing(input_dict, output_dict):
      return

    model_export = artifact_utils.get_single_instance(
        input_dict['model_export'])
    model_export_uri = model_export.uri
    model_blessing_uri = artifact_utils.get_single_uri(
        input_dict['model_blessing'])
    model_push = artifact_utils.get_single_instance(output_dict['model_push'])

    if not tf.io.gfile.exists(os.path.join(model_blessing_uri, 'BLESSED')):
      model_push.set_int_custom_property('pushed', 0)
      absl.logging.info('Model on %s was not blessed', model_blessing_uri)
      return

    custom_config = exec_properties.get('custom_config', {})
    bigquery_serving_args = custom_config.get('bigquery_serving_args', None)
    # if configuration is missing error out
    if bigquery_serving_args is None:
      raise RuntimeError('Big Query ML configuration was not provided')

    bg_model_uri = '`{}`.`{}`.`{}`'.format(
        bigquery_serving_args['project_id'],
        bigquery_serving_args['bq_dataset_id'],
        bigquery_serving_args['model_name'])

    # Deploy the model.
    model_path = path_utils.serving_model_path(model_export_uri)

    absl.logging.info(
        'Deploying the model to BigQuery ML for serving: {}'.format(
            bigquery_serving_args))

    query = ("""
      CREATE OR REPLACE MODEL {}
      OPTIONS (model_type='tensorflow',
               model_path='{}')""".format(bg_model_uri, model_path))

    client = bigquery.Client()

    try:
      query_job = client.query(query)
      query_job.result()  # Waits for the query to finish
    except Exception as e:
      raise RuntimeError('BigQuery ML Push failed: {}'.format(e))

    absl.logging.info('Successfully deployed model {} serving from {}'.format(
        bg_model_uri, model_path))

    # Setting the push_destination to bigquery uri
    model_push.set_int_custom_property('pushed', 1)
    model_push.set_string_custom_property('pushed_model', bg_model_uri)
