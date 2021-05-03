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
"""TFX InfraValidator component definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional

from tfx import types
from tfx.components.infra_validator import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_driver
from tfx.dsl.components.base import executor_spec
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class InfraValidator(base_component.BaseComponent):
  """A TFX component to validate the model against the serving infrastructure.

  An infra validation is done by loading the model to the exactly same serving
  binary that is used in production, and additionaly sending some requests to
  the model server. Such requests can be specified from Examples artifact.

  ## Examples

  Full example using TensorFlowServing binary running on local docker.

  ```
  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=test_example_gen.outputs['examples'],
      serving_spec=ServingSpec(
          tensorflow_serving=TensorFlowServing(  # Using TF Serving.
              tags=['latest']
          ),
          local_docker=LocalDockerConfig(),  # Running on local docker.
      ),
      validation_spec=ValidationSpec(
          max_loading_time_seconds=60,
          num_tries=5,
      ),
      request_spec=RequestSpec(
          tensorflow_serving=TensorFlowServingRequestSpec(),
          num_examples=1,
      )
  )
  ```

  Minimal example when running on Kubernetes.

  ```
  infra_validator = InfraValidator(
      model=trainer.outputs['model'],
      examples=test_example_gen.outputs['examples'],
      serving_spec=ServingSpec(
          tensorflow_serving=TensorFlowServing(
              tags=['latest']
          ),
          kubernetes=KubernetesConfig(),  # Running on Kubernetes.
      ),
  )
  ```

  Component `outputs` contains:
   - `blessing`: Channel of type `standard_artifacts.InfraBlessing` that
                 contains the validation result.
  """

  SPEC_CLASS = standard_component_specs.InfraValidatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(
      self,
      model: types.Channel,
      serving_spec: infra_validator_pb2.ServingSpec,
      examples: Optional[types.Channel] = None,
      request_spec: Optional[infra_validator_pb2.RequestSpec] = None,
      validation_spec: Optional[infra_validator_pb2.ValidationSpec] = None):
    """Construct a InfraValidator component.

    Args:
      model: A `Channel` of `ModelExportPath` type, usually produced by
        [Trainer](https://www.tensorflow.org/tfx/guide/trainer) component.
        _required_
      serving_spec: A `ServingSpec` configuration about serving binary and
        test platform config to launch model server for validation. _required_
      examples: A `Channel` of `ExamplesPath` type, usually produced by
        [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) component.
        If not specified, InfraValidator does not issue requests for validation.

      request_spec: Optional `RequestSpec` configuration about making requests
        from `examples` input. If not specified, InfraValidator does not issue
        requests for validation.
      validation_spec: Optional `ValidationSpec` configuration.
    """
    blessing = types.Channel(type=standard_artifacts.InfraBlessing)
    spec = standard_component_specs.InfraValidatorSpec(
        model=model,
        examples=examples,
        blessing=blessing,
        serving_spec=serving_spec,
        validation_spec=validation_spec,
        request_spec=request_spec
    )
    super(InfraValidator, self).__init__(spec=spec)
