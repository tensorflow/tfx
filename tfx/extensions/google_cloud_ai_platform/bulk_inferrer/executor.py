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
"""BulkInferrer executor for Cloud AI platform."""

import hashlib
import re
from typing import Any, Dict, List

from absl import logging
from google.api_core import client_options
from googleapiclient import discovery
import tensorflow as tf
from tfx import types
from tfx.components.bulk_inferrer import executor as bulk_inferrer_executor
from tfx.components.util import model_utils
from tfx.extensions.google_cloud_ai_platform import constants
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.utils import doc_controls
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils
from tfx.utils import telemetry_utils
from tfx_bsl.public.proto import model_spec_pb2

from tensorflow.python.saved_model import loader_impl  # pylint:disable=g-direct-tensorflow-import
# TODO(b/140306674): Stop using the internal TF API.

_CLOUD_PUSH_DESTINATION_RE = re.compile(
    r'^projects\/([^\/]+)\/models\/([^\/]+)\/versions\/([^\/]+)$')
_CLOUD_PUSH_DESTINATION_RE_DEFAULT_VERSION = re.compile(
    r'^projects\/([^\/]+)\/models\/([^\/]+)$')

# We define the following aliases of Any because the actual types are not
# public.
_SignatureDef = Any

# Keys to the items in custom_config passed as a part of exec_properties.
SERVING_ARGS_KEY = doc_controls.documented(
    obj='ai_platform_serving_args',
    doc='Keys to the items in custom_config of Bulk Inferrer for passing bulk'
    'inferrer args to AI Platform.')
# Keys for custom_config.
_CUSTOM_CONFIG_KEY = 'custom_config'


class Executor(bulk_inferrer_executor.Executor):
  """Bulk inferer executor for inference on AI Platform."""

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    """Runs batch inference on a given model with given input examples.

    This function creates a new model (if necessary) and a new model version
    before inference, and cleans up resources after inference. It provides
    re-executability as it cleans up (only) the model resources that are created
    during the process even inference job failed.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
        - model: exported model.
        - model_blessing: model blessing result
      output_dict: Output dict from output key to a list of Artifacts.
        - output: bulk inference results.
      exec_properties: A dict of execution properties.
        - data_spec: JSON string of bulk_inferrer_pb2.DataSpec instance.
        - custom_config: custom_config.ai_platform_serving_args need to contain
          the serving job parameters sent to Google Cloud AI Platform. For the
          full set of parameters, refer to
          https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if output_dict.get('inference_result'):
      inference_result = artifact_utils.get_single_instance(
          output_dict['inference_result'])
    else:
      inference_result = None
    if output_dict.get('output_examples'):
      output_examples = artifact_utils.get_single_instance(
          output_dict['output_examples'])
    else:
      output_examples = None

    if 'examples' not in input_dict:
      raise ValueError('`examples` is missing in input dict.')
    if 'model' not in input_dict:
      raise ValueError('Input models are not valid, model '
                       'need to be specified.')
    if 'model_blessing' in input_dict:
      model_blessing = artifact_utils.get_single_instance(
          input_dict['model_blessing'])
      if not model_utils.is_model_blessed(model_blessing):
        logging.info('Model on %s was not blessed', model_blessing.uri)
        return
    else:
      logging.info('Model blessing is not provided, exported model will be '
                   'used.')
    if _CUSTOM_CONFIG_KEY not in exec_properties:
      raise ValueError('Input exec properties are not valid, {} '
                       'need to be specified.'.format(_CUSTOM_CONFIG_KEY))

    custom_config = json_utils.loads(
        exec_properties.get(_CUSTOM_CONFIG_KEY, 'null'))
    if custom_config is not None and not isinstance(custom_config, Dict):
      raise ValueError('custom_config in execution properties needs to be a '
                       'dict.')
    ai_platform_serving_args = custom_config.get(SERVING_ARGS_KEY)
    if not ai_platform_serving_args:
      raise ValueError(
          '`ai_platform_serving_args` is missing in `custom_config`')
    service_name, api_version = runner.get_service_name_and_api_version(
        ai_platform_serving_args)
    executor_class_path = '%s.%s' % (self.__class__.__module__,
                                     self.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()
    model = artifact_utils.get_single_instance(input_dict['model'])
    model_path = path_utils.serving_model_path(
        model.uri, path_utils.is_old_model_artifact(model))
    logging.info('Use exported model from %s.', model_path)
    # Use model artifact uri to generate model version to guarantee the
    # 1:1 mapping from model version to model.
    model_version = 'version_' + hashlib.sha256(model.uri.encode()).hexdigest()
    inference_spec = self._get_inference_spec(model_path, model_version,
                                              ai_platform_serving_args)
    data_spec = bulk_inferrer_pb2.DataSpec()
    proto_utils.json_to_proto(exec_properties['data_spec'], data_spec)
    output_example_spec = bulk_inferrer_pb2.OutputExampleSpec()
    if exec_properties.get('output_example_spec'):
      proto_utils.json_to_proto(exec_properties['output_example_spec'],
                                output_example_spec)
    endpoint = custom_config.get(constants.ENDPOINT_ARGS_KEY)
    if endpoint and 'regions' in ai_platform_serving_args:
      raise ValueError(
          '`endpoint` and `ai_platform_serving_args.regions` cannot be set simultaneously'
      )
    api = discovery.build(
        service_name,
        api_version,
        requestBuilder=telemetry_utils.TFXHttpRequest,
        client_options=client_options.ClientOptions(api_endpoint=endpoint),
    )
    new_model_endpoint_created = False
    try:
      new_model_endpoint_created = runner.create_model_for_aip_prediction_if_not_exist(
          job_labels, ai_platform_serving_args, api)
      runner.deploy_model_for_aip_prediction(
          serving_path=model_path,
          model_version_name=model_version,
          ai_platform_serving_args=ai_platform_serving_args,
          api=api,
          labels=job_labels,
          skip_model_endpoint_creation=True,
          set_default=False,
      )
      self._run_model_inference(data_spec, output_example_spec,
                                input_dict['examples'], output_examples,
                                inference_result, inference_spec)
    except Exception as e:
      logging.error('Error in executing CloudAIBulkInferrerComponent: %s',
                    str(e))
      raise
    finally:
      # Guarantee newly created resources are cleaned up even if the inference
      # job failed.

      # Clean up the newly deployed model.
      runner.delete_model_from_aip_if_exists(
          model_version_name=model_version,
          ai_platform_serving_args=ai_platform_serving_args,
          api=api,
          delete_model_endpoint=new_model_endpoint_created)

  def _get_inference_spec(
      self, model_path: str, model_version: str,
      ai_platform_serving_args: Dict[str, Any]
  ) -> model_spec_pb2.InferenceSpecType:
    if 'project_id' not in ai_platform_serving_args:
      raise ValueError('`project_id` is missing in `ai_platform_serving_args`')
    project_id = ai_platform_serving_args['project_id']
    if 'model_name' not in ai_platform_serving_args:
      raise ValueError('`model_name` is missing in `ai_platform_serving_args`')
    model_name = ai_platform_serving_args['model_name']
    ai_platform_prediction_model_spec = (
        model_spec_pb2.AIPlatformPredictionModelSpec(
            project_id=project_id,
            model_name=model_name,
            version_name=model_version))
    model_signature = self._get_model_signature(model_path)
    if (len(model_signature.inputs) == 1 and list(
        model_signature.inputs.values())[0].dtype == tf.string.as_datatype_enum
       ):
      ai_platform_prediction_model_spec.use_serialization_config = True
    logging.info(
        'Using hosted model on Cloud AI platform, model_name: %s,'
        'model_version: %s.', model_name, model_version)
    result = model_spec_pb2.InferenceSpecType()
    result.ai_platform_prediction_model_spec.CopyFrom(
        ai_platform_prediction_model_spec)
    return result

  def _get_model_signature(self, model_path: str) -> _SignatureDef:
    """Returns a model signature."""

    saved_model_pb = loader_impl.parse_saved_model(model_path)
    meta_graph_def = None
    for graph_def in saved_model_pb.meta_graphs:
      if graph_def.meta_info_def.tags == [
          tf.compat.v1.saved_model.tag_constants.SERVING
      ]:
        meta_graph_def = graph_def
    if not meta_graph_def:
      raise RuntimeError('Tag tf.compat.v1.saved_model.tag_constants.SERVING'
                         ' does not exist in saved model: %s. This is required'
                         ' for remote inference.' % model_path)
    if tf.saved_model.PREDICT_METHOD_NAME in meta_graph_def.signature_def:
      return meta_graph_def.signature_def[tf.saved_model.PREDICT_METHOD_NAME]
    if (tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY in
        meta_graph_def.signature_def):
      return meta_graph_def.signature_def[
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    raise RuntimeError(
        'Cannot find serving signature in saved model: %s,'
        ' tf.saved_model.PREDICT_METHOD_NAME or '
        ' tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY is needed.' %
        model_path)
