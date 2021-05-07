# Lint as: python2, python3
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
"""TFX Pusher component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from absl import logging
from tfx import types
from tfx.components.pusher import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import PusherSpec
from tfx.utils import json_utils


# TODO(b/133845381): Investigate other ways to keep push destination converged.
class Pusher(base_component.BaseComponent):
  """A TFX component to push validated TensorFlow models to a model serving platform.

  The `Pusher` component can be used to push an validated SavedModel from output
  of the [Trainer component](https://www.tensorflow.org/tfx/guide/trainer) to
  [TensorFlow Serving](https://www.tensorflow.org/tfx/serving).  The Pusher
  will check the validation results from the [Evaluator
  component](https://www.tensorflow.org/tfx/guide/evaluator) and [InfraValidator
  component](https://www.tensorflow.org/tfx/guide/infra_validator)
  before deploying the model.  If the model has not been blessed, then the model
  will not be pushed.

  *Note:* The executor for this component can be overriden to enable the model
  to be pushed to other serving platforms than tf.serving.  The [Cloud AI
  Platform custom
  executor](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_ai_platform/pusher)
  provides an example how to implement this.

  ## Example
  ```
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
  ```

  Component `outputs` contains:
   - `pushed_model`: Channel of type `standard_artifacts.PushedModel` with
                     result of push.
  """

  SPEC_CLASS = PusherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      model: Optional[types.Channel] = None,
      model_blessing: Optional[types.Channel] = None,
      infra_blessing: Optional[types.Channel] = None,
      push_destination: Optional[Union[pusher_pb2.PushDestination,
                                       Dict[Text, Any]]] = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
    """Construct a Pusher component.

    Args:
      model: An optional Channel of type `standard_artifacts.Model`, usually
        produced by a Trainer component.
      model_blessing: An optional Channel of type
        `standard_artifacts.ModelBlessing`, usually produced from an Evaluator
        component.
      infra_blessing: An optional Channel of type
        `standard_artifacts.InfraBlessing`, usually produced from an
        InfraValidator component.
      push_destination: A pusher_pb2.PushDestination instance, providing info
        for tensorflow serving to load models. Optional if executor_class
        doesn't require push_destination. If any field is provided as a
        RuntimeParameter, push_destination should be constructed as a dict with
        the same field names as PushDestination proto message.
      custom_config: A dict which contains the deployment job parameters to be
        passed to Cloud platforms.
      custom_executor_spec: Optional custom executor spec. This is experimental
        and is subject to change in the future.
    """
    pushed_model = types.Channel(type=standard_artifacts.PushedModel)
    if (push_destination is None and not custom_executor_spec and
        self.EXECUTOR_SPEC.executor_class == executor.Executor):
      raise ValueError('push_destination is required unless a '
                       'custom_executor_spec is supplied that does not require '
                       'it.')
    if custom_executor_spec:
      logging.warning('`custom_executor_spec` is going to be deprecated.')
    if model is None and infra_blessing is None:
      raise ValueError(
          'Either one of model or infra_blessing channel should be given. '
          'If infra_blessing is used in place of model, it must have been '
          'created with InfraValidator with RequestSpec.make_warmup = True. '
          'This cannot be checked during pipeline construction time but will '
          'raise runtime error if infra_blessing does not contain a model.')
    spec = PusherSpec(
        model=model,
        model_blessing=model_blessing,
        infra_blessing=infra_blessing,
        push_destination=push_destination,
        custom_config=json_utils.dumps(custom_config),
        pushed_model=pushed_model)
    super(Pusher, self).__init__(
        spec=spec, custom_executor_spec=custom_executor_spec)
