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
"""TFX Trainer component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.trainer import driver
from tfx.components.trainer import executor
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TrainerSpec


class Trainer(base_component.BaseComponent):
  """A TFX component to train a TensorFlow model.

  The Trainer component is used to train and eval a model using given inputs and
  a user-supplied estimator.  This component includes a custom driver to
  optionally grab previous model to warm start from.

  ## Providing an estimator
  The TFX executor will use the estimator provided in the `module_file` file
  to train the model.  The Trainer executor will look specifically for the
  `trainer_fn()` function within that file.  Before training, the executor will
  call that function expecting the following returned as a dictionary:

    - estimator: The
    [estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
    to be used by TensorFlow to train the model.
    - train_spec: The
    [configuration](https://www.tensorflow.org/api_docs/python/tf/estimator/TrainSpec)
    to be used by the "train" part of the TensorFlow `train_and_evaluate()`
    call.
    - eval_spec: The
    [configuration](https://www.tensorflow.org/api_docs/python/tf/estimator/EvalSpec)
    to be used by the "eval" part of the TensorFlow `train_and_evaluate()` call.
    - eval_input_receiver_fn: The
    [configuration](https://www.tensorflow.org/tfx/model_analysis/get_started#modify_an_existing_model)
    to be used
    by the [ModelValidator](https://www.tensorflow.org/tfx/guide/modelval)
    component when validating the model.

  An example of `trainer_fn()` can be found in the [user-supplied
  code]((https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils.py))
  of the TFX Chicago Taxi pipeline example.

  *Note:* The default executor for this component trains locally.  This can be
  overriden to enable the model to be trained on other platforms.  The [Cloud AI
  Platform custom
  executor](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_ai_platform/trainer)
  provides an example how to implement this.

  Please see https://www.tensorflow.org/guide/estimators for more details.

  ## Example 1: Training locally
  ```
  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  ```

  ## Example 2: Training through a cloud provider
  ```
  # Train using Google Cloud AI Platform.
  trainer = Trainer(
      executor_class=ai_platform_trainer_executor.Executor,
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))
  ```
  """

  SPEC_CLASS = TrainerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)
  DRIVER_CLASS = driver.Driver

  def __init__(
      self,
      examples: types.Channel = None,
      transformed_examples: Optional[types.Channel] = None,
      transform_graph: Optional[types.Channel] = None,
      schema: types.Channel = None,
      module_file: Optional[Text] = None,
      trainer_fn: Optional[Text] = None,
      train_args: trainer_pb2.TrainArgs = None,
      eval_args: trainer_pb2.EvalArgs = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      output: Optional[types.Channel] = None,
      transform_output: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a Trainer component.

    Args:
      examples: A Channel of 'ExamplesPath' type, serving as the source of
        examples that are used in training (required). May be raw or
        transformed.
      transformed_examples: Deprecated field. Please set 'examples' instead.
      transform_graph: An optional Channel of 'TransformPath' type, serving as
        the input transform graph if present.
      schema:  A Channel of 'SchemaPath' type, serving as the schema of training
        and eval data.
      module_file: A path to python module file containing UDF model definition.
        The module_file must implement a function named `trainer_fn` at its
        top level. The function must have the following signature.

        def trainer_fn(tf.contrib.training.HParams,
                       tensorflow_metadata.proto.v0.schema_pb2) -> Dict:
          ...

        where the returned Dict has the following key-values.
          'estimator': an instance of tf.estimator.Estimator
          'train_spec': an instance of tf.estimator.TrainSpec
          'eval_spec': an instance of tf.estimator.EvalSpec
          'eval_input_receiver_fn': an instance of tfma.export.EvalInputReceiver

        Exactly one of 'module_file' or 'trainer_fn' must be supplied.
      trainer_fn:  A python path to UDF model definition function. See
        'module_file' for the required signature of the UDF.
        Exactly one of 'module_file' or 'trainer_fn' must be supplied.
      train_args: A trainer_pb2.TrainArgs instance, containing args used for
        training. Current only num_steps is available.
      eval_args: A trainer_pb2.EvalArgs instance, containing args used for eval.
        Current only num_steps is available.
      custom_config: A dict which contains the training job parameters to be
        passed to Google Cloud ML Engine.  For the full set of parameters
        supported by Google Cloud ML Engine, refer to
        https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
      custom_executor_spec: Optional custom executor spec.
      output: Optional 'ModelExportPath' channel for result of exported models.
      transform_output: Backwards compatibility alias for the 'transform_graph'
        argument.
      instance_name: Optional unique instance name. Necessary iff multiple
        Trainer components are declared in the same pipeline.

    Raises:
      ValueError:
        - When both or neither of 'module_file' and 'trainer_fn' is supplied.
        - When both or neither of 'examples' and 'transformed_examples'
            is supplied.
        - When 'transformed_examples' is supplied but 'transform_output'
            is not supplied.
    """
    transform_graph = transform_graph or transform_output
    if bool(module_file) == bool(trainer_fn):
      raise ValueError(
          "Exactly one of 'module_file' or 'trainer_fn' must be supplied")

    if bool(examples) == bool(transformed_examples):
      raise ValueError(
          "Exactly one of 'example' or 'transformed_example' must be supplied.")

    if transformed_examples and not transform_graph:
      raise ValueError("If 'transformed_examples' is supplied, "
                       "'transform_graph' must be supplied too.")
    examples = examples or transformed_examples
    output = output or types.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    spec = TrainerSpec(
        examples=examples,
        transform_output=transform_graph,
        schema=schema,
        train_args=train_args,
        eval_args=eval_args,
        module_file=module_file,
        trainer_fn=trainer_fn,
        custom_config=custom_config,
        output=output)
    super(Trainer, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
        instance_name=instance_name)
