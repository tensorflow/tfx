# Lint as: python2, python3
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
"""BulkInferrer component for Cloud AI platform."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from tfx import types
from tfx.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.bulk_inferrer import executor
from tfx.proto import bulk_inferrer_pb2
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import json_utils


class CloudAIBulkInferrerComponentSpec(types.ComponentSpec):
  """ComponentSpec for BulkInferrer component of Cloud AI platform."""

  PARAMETERS = {
      'data_spec':
          ExecutionParameter(type=bulk_inferrer_pb2.DataSpec, optional=True),
      'output_example_spec':
          ExecutionParameter(
              type=bulk_inferrer_pb2.OutputExampleSpec, optional=True),
      'custom_config':
          ExecutionParameter(type=(str, Text)),
  }
  INPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples),
      'model':
          ChannelParameter(type=standard_artifacts.Model),
      'model_blessing':
          ChannelParameter(
              type=standard_artifacts.ModelBlessing, optional=True),
  }
  OUTPUTS = {
      'inference_result':
          ChannelParameter(
              type=standard_artifacts.InferenceResult, optional=True),
      'output_examples':
          ChannelParameter(type=standard_artifacts.Examples, optional=True),
  }


class CloudAIBulkInferrerComponent(base_component.BaseComponent):
  """A Cloud AI component to do batch inference on a remote hosted model.

  BulkInferrer component will push a model to Google Cloud AI Platform,
  consume examples data, send request to the remote hosted model,
  and produces the inference results to an external location
  as PredictionLog proto. After inference, it will delete the model from
  Google Cloud AI Platform.

  TODO(b/155325467): Creates a end-to-end test for this component.

  Component `outputs` contains:
   - `inference_result`: Channel of type `standard_artifacts.InferenceResult`
                         to store the inference results.
   - `output_examples`: Channel of type `standard_artifacts.Examples`
                        to store the output examples.
  """

  SPEC_CLASS = CloudAIBulkInferrerComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      model: Optional[types.Channel] = None,
      model_blessing: Optional[types.Channel] = None,
      data_spec: Optional[Union[bulk_inferrer_pb2.DataSpec, Dict[Text,
                                                                 Any]]] = None,
      output_example_spec: Optional[Union[bulk_inferrer_pb2.OutputExampleSpec,
                                          Dict[Text, Any]]] = None,
      custom_config: Dict[Text, Any] = None):
    """Construct an BulkInferrer component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, usually
        produced by an ExampleGen component. _required_
      model: A Channel of type `standard_artifacts.Model`, usually produced by
        a Trainer component.
      model_blessing: A Channel of type `standard_artifacts.ModelBlessing`,
        usually produced by a ModelValidator component.
      data_spec: bulk_inferrer_pb2.DataSpec instance that describes data
        selection. If any field is provided as a RuntimeParameter, data_spec
        should be constructed as a dict with the same field names as DataSpec
        proto message.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance, specify
        if you want BulkInferrer to output examples instead of inference result.
        If any field is provided as a RuntimeParameter, output_example_spec
        should be constructed as a dict with the same field names as
        OutputExampleSpec proto message.
      custom_config: A dict which contains the deployment job parameters to be
        passed to Google Cloud AI Platform.
        custom_config.ai_platform_serving_args need to contain the serving job
        parameters. For the full set of parameters, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

    Raises:
      ValueError: Must not specify inference_result or output_examples depends
        on whether output_example_spec is set or not.
    """
    if output_example_spec:
      output_examples = types.Channel(type=standard_artifacts.Examples)
      inference_result = None
    else:
      inference_result = types.Channel(type=standard_artifacts.InferenceResult)
      output_examples = None

    spec = CloudAIBulkInferrerComponentSpec(
        examples=examples,
        model=model,
        model_blessing=model_blessing,
        data_spec=data_spec or bulk_inferrer_pb2.DataSpec(),
        output_example_spec=output_example_spec,
        custom_config=json_utils.dumps(custom_config),
        inference_result=inference_result,
        output_examples=output_examples)
    super(CloudAIBulkInferrerComponent, self).__init__(spec=spec)
