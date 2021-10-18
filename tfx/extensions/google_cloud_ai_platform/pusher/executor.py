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

import time
from typing import Any, Dict, List

from google.api_core import client_options
from googleapiclient import discovery
from tfx import types
from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.extensions.google_cloud_ai_platform import constants
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import deprecation_utils
from tfx.utils import io_utils
from tfx.utils import json_utils
from tfx.utils import telemetry_utils

# Keys for custom_config.
_CUSTOM_CONFIG_KEY = 'custom_config'


class Executor(tfx_pusher_executor.Executor):
  """Deploy a model to Google Cloud AI Platform serving."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]):
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
          by
          - Google Cloud AI Platform, refer to
          https://cloud.google.com/ml-engine/reference/rest/v1/projects.models.versions#Version.
          - Google Cloud Vertex AI, refer to
          https://googleapis.dev/python/aiplatform/latest/aiplatform.html?highlight=deploy#google.cloud.aiplatform.Model.deploy
        - endpoint: Optional endpoint override.
          - For Google Cloud AI Platform, this should be in format of
            `https://[region]-ml.googleapis.com`. Default to global endpoint if
            not set. Using regional endpoint is recommended by Cloud AI
            Platform. When set, 'regions' key in ai_platform_serving_args cannot
            be set. For more details, please see
            https://cloud.google.com/ai-platform/prediction/docs/regional-endpoints#using_regional_endpoints
          - For Google Cloud Vertex AI, this should be just be `region` (e.g.
            'us-central1'). For available regions, please see
            https://cloud.google.com/vertex-ai/docs/general/locations

    Raises:
      ValueError:
        If ai_platform_serving_args is not in exec_properties.custom_config.
        If Serving model path does not start with gs://.
        If 'endpoint' and 'regions' are set simultaneously.
      RuntimeError: if the Google Cloud AI Platform training job failed.
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')
    ai_platform_serving_args = custom_config.get(constants.SERVING_ARGS_KEY)
    if not ai_platform_serving_args:
      raise ValueError(
          '\'ai_platform_serving_args\' is missing in \'custom_config\'')
    model_push = artifact_utils.get_single_instance(
        output_dict[standard_component_specs.PUSHED_MODEL_KEY])
    if not self.CheckBlessing(input_dict):
      self._MarkNotPushed(model_push)
      return

    # Deploy the model.
    io_utils.copy_dir(src=self.GetModelPath(input_dict), dst=model_push.uri)
    model_path = model_push.uri

    executor_class_path = '%s.%s' % (self.__class__.__module__,
                                     self.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()

    enable_vertex = custom_config.get(constants.ENABLE_VERTEX_KEY)
    if enable_vertex:
      if custom_config.get(constants.ENDPOINT_ARGS_KEY):
        deprecation_utils.warn_deprecated(
            '\'endpoint\' is deprecated. Please use'
            '\'ai_platform_vertex_region\' instead.'
        )
      if 'regions' in ai_platform_serving_args:
        deprecation_utils.warn_deprecated(
            '\'ai_platform_serving_args.regions\' is deprecated. Please use'
            '\'ai_platform_vertex_region\' instead.'
        )
      endpoint_region = custom_config.get(constants.VERTEX_REGION_KEY)
      # TODO(jjong): Introduce Versioning.
      # Note that we're adding "v" prefix as Cloud AI Prediction only allows the
      # version name that starts with letters, and contains letters, digits,
      # underscore only.
      model_name = 'v{}'.format(int(time.time()))
      container_image_uri = custom_config.get(
          constants.VERTEX_CONTAINER_IMAGE_URI_KEY)

      pushed_model_path = runner.deploy_model_for_aip_prediction(
          serving_container_image_uri=container_image_uri,
          model_version_name=model_name,
          ai_platform_serving_args=ai_platform_serving_args,
          endpoint_region=endpoint_region,
          labels=job_labels,
          serving_path=model_path,
          enable_vertex=True,
      )

      self._MarkPushed(
          model_push,
          pushed_destination=pushed_model_path)

    else:
      endpoint = custom_config.get(constants.ENDPOINT_ARGS_KEY)
      if endpoint and 'regions' in ai_platform_serving_args:
        raise ValueError(
            '\'endpoint\' and \'ai_platform_serving_args.regions\' cannot be set simultaneously'
        )
      # TODO(jjong): Introduce Versioning.
      # Note that we're adding "v" prefix as Cloud AI Prediction only allows the
      # version name that starts with letters, and contains letters, digits,
      # underscore only.
      model_version = 'v{}'.format(int(time.time()))
      endpoint = endpoint or runner.DEFAULT_ENDPOINT
      service_name, api_version = runner.get_service_name_and_api_version(
          ai_platform_serving_args)
      api = discovery.build(
          service_name,
          api_version,
          requestBuilder=telemetry_utils.TFXHttpRequest,
          client_options=client_options.ClientOptions(api_endpoint=endpoint),
      )
      pushed_model_version_path = runner.deploy_model_for_aip_prediction(
          serving_path=model_path,
          model_version_name=model_version,
          ai_platform_serving_args=ai_platform_serving_args,
          api=api,
          labels=job_labels,
      )

      self._MarkPushed(
          model_push,
          pushed_destination=pushed_model_version_path,
          pushed_version=model_version)
