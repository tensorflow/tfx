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
"""Custom executor to push TFX model to AI Platform."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from typing import Any, Dict, List, Text

from google.api_core import client_options  # pylint: disable=unused-import
from googleapiclient import discovery
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx.utils import telemetry_utils

# Google Cloud AI Platform's ModelVersion resource path format.
# https://cloud.google.com/ai-platform/prediction/docs/reference/rest/v1/projects.models.versions/get
_CAIP_MODEL_VERSION_PATH_FORMAT = (
    'projects/{project_id}/models/{model}/versions/{version}')

# Keys to the items in custom_config passed as a part of exec_properties.
SERVING_ARGS_KEY = 'ai_platform_serving_args'
ENDPOINT_ARGS_KEY = 'endpoint'
# Keys for custom_config.
_CUSTOM_CONFIG_KEY = 'custom_config'


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to Google Cloud AI Platform serving."""

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
        tfx.components.Pusher.executor.  The following keys in `custom_config`
        are consumed by this class:
        - ai_platform_serving_args: For the full set of parameters supported
          by Google Cloud AI Platform, refer to
          https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version.
        - endpoint: Optional endpoint override. Should be in format of
          `https://[region]-ml.googleapis.com`. Default to global endpoint if
          not set. Using regional endpoint is recommended by Cloud AI Platform.
          When set, 'regions' key in ai_platform_serving_args cannot be set.
          For more details, please see
          https://cloud.google.com/ai-platform/prediction/docs/regional-endpoints#using_regional_endpoints

    Raises:
      ValueError:
        If ai_platform_serving_args is not in exec_properties.custom_config.
        If Serving model path does not start with gs://.
        If 'endpoint' and 'regions' are set simultanuously.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')
    ai_platform_serving_args = custom_config.get(SERVING_ARGS_KEY)
    if not ai_platform_serving_args:
      raise ValueError(
          '\'ai_platform_serving_args\' is missing in \'custom_config\'')
    endpoint = custom_config.get(ENDPOINT_ARGS_KEY)
    if endpoint and 'regions' in ai_platform_serving_args:
      raise ValueError(
          '\'endpoint\' and \'ai_platform_serving_args.regions\' cannot be set simultanuously'
      )

    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY])
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return

    model_export = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.MODEL_KEY])

    service_name, api_version = runner.get_service_name_and_api_version(
        ai_platform_serving_args)
    # Deploy the model.
    io_utils.copy_dir(
        src=path_utils.serving_model_path(
            model_export.uri, path_utils.is_old_model_artifact(model_export)),
        dst=model_push.uri)
    model_path = model_push.uri
    # TODO(jjong): Introduce Versioning.
    # Note that we're adding "v" prefix as Cloud AI Prediction only allows the
    # version name that starts with letters, and contains letters, digits,
    # underscore only.
    model_version = 'v{}'.format(int(time.time()))
    executor_class_path = '%s.%s' % (self.__class__.__module__,
                                     self.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.get_labels_dict()
    endpoint = endpoint or runner.DEFAULT_ENDPOINT
    api = discovery.build(
        service_name,
        api_version,
        client_options=client_options.ClientOptions(api_endpoint=endpoint),
    )
    runner.deploy_model_for_aip_prediction(
        api,
        model_path,
        model_version,
        ai_platform_serving_args,
        job_labels,
    )

    self._MarkPushed(
        model_push,
        pushed_destination=_CAIP_MODEL_VERSION_PATH_FORMAT.format(
            project_id=ai_platform_serving_args['project_id'],
            model=ai_platform_serving_args['model_name'],
            version=model_version),
        pushed_version=model_version)
