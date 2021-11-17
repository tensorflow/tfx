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

from typing import Optional, Union

from tfx import types
from tfx.components.bulk_inferrer import executor
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import bulk_inferrer_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class BulkInferrer(base_beam_component.BaseBeamComponent):
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

  Component `outputs` contains:
   - `inference_result`: Channel of type `standard_artifacts.InferenceResult`
                         to store the inference results.
   - `output_examples`: Channel of type `standard_artifacts.Examples`
                        to store the output examples. This is optional
                        controlled by `output_example_spec`.

  See [the BulkInferrer
  guide](https://www.tensorflow.org/tfx/guide/bulkinferrer) for more details.
  """

  SPEC_CLASS = standard_component_specs.BulkInferrerSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel,
      model: Optional[types.Channel] = None,
      model_blessing: Optional[types.Channel] = None,
      data_spec: Optional[Union[bulk_inferrer_pb2.DataSpec,
                                data_types.RuntimeParameter]] = None,
      model_spec: Optional[Union[bulk_inferrer_pb2.ModelSpec,
                                 data_types.RuntimeParameter]] = None,
      output_example_spec: Optional[Union[bulk_inferrer_pb2.OutputExampleSpec,
                                          data_types.RuntimeParameter]] = None):
    """Construct an BulkInferrer component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, usually
        produced by an ExampleGen component. _required_
      model: A Channel of type `standard_artifacts.Model`, usually produced by a
        Trainer component.
      model_blessing: A Channel of type `standard_artifacts.ModelBlessing`,
        usually produced by a ModelValidator component.
      data_spec: bulk_inferrer_pb2.DataSpec instance that describes data
        selection.
      model_spec: bulk_inferrer_pb2.ModelSpec instance that describes model
        specification.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance, specify
        if you want BulkInferrer to output examples instead of inference result.
    """
    if output_example_spec:
      output_examples = types.Channel(type=standard_artifacts.Examples)
      inference_result = None
    else:
      inference_result = types.Channel(type=standard_artifacts.InferenceResult)
      output_examples = None

    spec = standard_component_specs.BulkInferrerSpec(
        examples=examples,
        model=model,
        model_blessing=model_blessing,
        data_spec=data_spec or bulk_inferrer_pb2.DataSpec(),
        model_spec=model_spec or bulk_inferrer_pb2.ModelSpec(),
        output_example_spec=output_example_spec,
        inference_result=inference_result,
        output_examples=output_examples)
    super().__init__(spec=spec)
