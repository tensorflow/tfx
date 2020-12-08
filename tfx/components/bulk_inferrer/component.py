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
"""TFX BulkInferrer component definition."""

from typing import Any, Dict, Optional, Text, Union

from tfx import types
from tfx.components.bulk_inferrer import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import bulk_inferrer_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import BulkInferrerSpec


class BulkInferrer(base_component.BaseComponent):
  """A TFX component to do batch inference on a model with unlabelled examples.

  BulkInferrer consumes examples data and a model, and produces the inference
  results to an external location as PredictionLog proto.

  BulkInferrer will infer on validated model.

  ## Example
  ```
    # Uses BulkInferrer to inference on examples.
    bulk_inferrer = BulkInferrer(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'])
  ```
  """

  SPEC_CLASS = BulkInferrerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      model: Optional[types.Channel] = None,
      model_blessing: Optional[types.Channel] = None,
      data_spec: Optional[Union[bulk_inferrer_pb2.DataSpec, Dict[Text,
                                                                 Any]]] = None,
      model_spec: Optional[Union[bulk_inferrer_pb2.ModelSpec,
                                 Dict[Text, Any]]] = None,
      output_example_spec: Optional[Union[bulk_inferrer_pb2.OutputExampleSpec,
                                          Dict[Text, Any]]] = None,
      inference_result: Optional[types.Channel] = None,
      output_examples: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
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
      model_spec: bulk_inferrer_pb2.ModelSpec instance that describes model
        specification. If any field is provided as a RuntimeParameter,
        model_spec should be constructed as a dict with the same field names as
        ModelSpec proto message.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance, specify
        if you want BulkInferrer to output examples instead of inference result.
        If any field is provided as a RuntimeParameter, output_example_spec
        should be constructed as a dict with the same field names as
        OutputExampleSpec proto message.
      inference_result: Channel of type `standard_artifacts.InferenceResult`
        to store the inference results, must not be specified when
        output_example_spec is set.
      output_examples: Channel of type `standard_artifacts.Examples`
        to store the output examples, must not be specified when
        output_example_spec is unset. Check output_example_spec for details.
      instance_name: Optional name assigned to this specific instance of
        BulkInferrer. Required only if multiple BulkInferrer components are
        declared in the same pipeline.

    Raises:
      ValueError: Must not specify inference_result or output_examples depends
        on whether output_example_spec is set or not.
    """
    if output_example_spec:
      if inference_result:
        raise ValueError(
            'Must not specify inference_result when output_example_spec is set.'
        )
      output_examples = output_examples or types.Channel(
          type=standard_artifacts.Examples)
    else:
      if output_examples:
        raise ValueError(
            'Must not specify output_examples when output_example_spec is unset.'
        )
      inference_result = inference_result or types.Channel(
          type=standard_artifacts.InferenceResult)

    spec = BulkInferrerSpec(
        examples=examples,
        model=model,
        model_blessing=model_blessing,
        data_spec=data_spec or bulk_inferrer_pb2.DataSpec(),
        model_spec=model_spec or bulk_inferrer_pb2.ModelSpec(),
        output_example_spec=output_example_spec,
        inference_result=inference_result,
        output_examples=output_examples)
    super(BulkInferrer, self).__init__(spec=spec, instance_name=instance_name)
