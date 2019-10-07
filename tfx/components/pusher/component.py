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

from typing import Any, Dict, Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.pusher import executor
from tfx.proto import pusher_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import PusherSpec


# TODO(b/133845381): Investigate other ways to keep push destination converged.
class Pusher(base_component.BaseComponent):
  """A TFX component to push validated TensorFlow models to a model serving platform.

  The `Pusher` component can be used to push an validated SavedModel from output
  of the [Trainer component](https://www.tensorflow.org/tfx/guide/trainer) to
  [TensorFlow Serving](https://www.tensorflow.org/tfx/serving).  The Pusher
  will check the validation results from the [ModelValidator
  component](https://www.tensorflow.org/tfx/guide/model_validator)
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
        model_blessing=model_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
  ```
  """

  SPEC_CLASS = PusherSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      model: types.Channel = None,
      model_blessing: types.Channel = None,
      push_destination: Optional[pusher_pb2.PushDestination] = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      model_push: Optional[types.Channel] = None,
      model_export: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a Pusher component.

    Args:
      model: A Channel of 'ModelExportPath' type, usually produced by
        Trainer component. Will be deprecated in the future for the `model`
        parameter.
      model_blessing: A Channel of 'ModelBlessingPath' type, usually produced by
        ModelValidator component. _required_
      push_destination: A pusher_pb2.PushDestination instance, providing info
        for tensorflow serving to load models. Optional if executor_class
        doesn't require push_destination.
      custom_config: A dict which contains the deployment job parameters to be
        passed to cloud-based training platforms.  The
        [Kubeflow
          example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_pipeline_kubeflow.py#L211)
          contains an example how this can be used by custom executors.
      custom_executor_spec: Optional custom executor spec.
      model_push: Optional output 'ModelPushPath' channel with result of push.
      model_export: Backwards compatibility alias for the 'model' argument.
      instance_name: Optional unique instance name. Necessary if multiple Pusher
        components are declared in the same pipeline.
    """
    model = model or model_export
    model_push = model_push or types.Channel(
        type=standard_artifacts.PushedModel,
        artifacts=[standard_artifacts.PushedModel()])
    if push_destination is None and not custom_executor_spec:
      raise ValueError('push_destination is required unless a '
                       'custom_executor_spec is supplied that does not require '
                       'it.')
    spec = PusherSpec(
        model_export=model,
        model_blessing=model_blessing,
        push_destination=push_destination,
        custom_config=custom_config,
        model_push=model_push)
    super(Pusher, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
        instance_name=instance_name)
