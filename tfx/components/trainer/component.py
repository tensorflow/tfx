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
"""TFX Trainer component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

import absl

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.trainer import executor
from tfx.orchestration import data_types
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import TrainerSpec
from tfx.utils import json_utils


# TODO(b/147702778): update when switch generic executor as default.
class Trainer(base_component.BaseComponent):
  """A TFX component to train a TensorFlow model.

  The Trainer component is used to train and eval a model using given inputs and
  a user-supplied estimator.

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
      train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=5000))
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
      train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=5000))
  ```
  """

  SPEC_CLASS = TrainerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      examples: types.Channel = None,
      transformed_examples: Optional[types.Channel] = None,
      transform_graph: Optional[types.Channel] = None,
      schema: types.Channel = None,
      base_model: Optional[types.Channel] = None,
      hyperparameters: Optional[types.Channel] = None,
      module_file: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      run_fn: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      # TODO(b/147702778): deprecate trainer_fn.
      trainer_fn: Optional[Union[Text, data_types.RuntimeParameter]] = None,
      train_args: Union[trainer_pb2.TrainArgs, Dict[Text, Any]] = None,
      eval_args: Union[trainer_pb2.EvalArgs, Dict[Text, Any]] = None,
      custom_config: Optional[Dict[Text, Any]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      output: Optional[types.Channel] = None,
      model_run: Optional[types.Channel] = None,
      transform_output: Optional[types.Channel] = None,
      instance_name: Optional[Text] = None):
    """Construct a Trainer component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as
        the source of examples used in training (required). May be raw or
        transformed.
      transformed_examples: Deprecated field. Please set 'examples' instead.
      transform_graph: An optional Channel of type
        `standard_artifacts.TransformGraph`, serving as the input transform
        graph if present.
      schema:  A Channel of type `standard_artifacts.Schema`, serving as the
        schema of training and eval data.
      base_model: A Channel of type `Model`, containing model that will be used
        for training. This can be used for warmstart, transfer learning or
        model ensembling.
      hyperparameters: A Channel of type `standard_artifacts.HyperParameters`,
        serving as the hyperparameters for training module. Tuner's output best
        hyperparameters can be feed into this.
      module_file: A path to python module file containing UDF model definition.

        For default executor, The module_file must implement a function named
        `trainer_fn` at its top level. The function must have the following
        signature.

        def trainer_fn(trainer.executor.TrainerFnArgs,
                       tensorflow_metadata.proto.v0.schema_pb2) -> Dict:
          ...

        where the returned Dict has the following key-values.
          'estimator': an instance of tf.estimator.Estimator
          'train_spec': an instance of tf.estimator.TrainSpec
          'eval_spec': an instance of tf.estimator.EvalSpec
          'eval_input_receiver_fn': an instance of
            tfma.export.EvalInputReceiver. Exactly one of 'module_file' or
            'trainer_fn' must be supplied.

        For generic executor, The module_file must implement a function named
        `run_fn` at its top level with function signature:
        `def run_fn(trainer.executor.TrainerFnArgs)`, and the trained model must
        be saved to TrainerFnArgs.serving_model_dir when execute this function.
      run_fn:  A python path to UDF model definition function for generic
        trainer. See 'module_file' for details. Exactly one of 'module_file' or
        'run_fn' must be supplied if Trainer uses GenericExecutor.
      trainer_fn:  A python path to UDF model definition function for estimator
        based trainer. See 'module_file' for the required signature of the UDF.
        Exactly one of 'module_file' or 'trainer_fn' must be supplied.
      train_args: A trainer_pb2.TrainArgs instance or a dict, containing args
        used for training. Currently only splits and num_steps are available. If
        it's provided as a dict and any field is a RuntimeParameter, it should
        have the same field names as a TrainArgs proto message. Default
        behavior (when splits is empty) is train on `train` split.
      eval_args: A trainer_pb2.EvalArgs instance or a dict, containing args
        used for evaluation. Currently only splits and num_steps are available.
        If it's provided as a dict and any field is a RuntimeParameter, it
        should have the same field names as a EvalArgs proto message. Default
        behavior (when splits is empty) is evaluate on `eval` split.
      custom_config: A dict which contains addtional training job parameters
        that will be passed into user module.
      custom_executor_spec: Optional custom executor spec.
      output: Optional `Model` channel for result of exported models.
      model_run: Optional `ModelRun` channel, as the working dir of models,
        can be used to output non-model related output (e.g., TensorBoard logs).
      transform_output: Backwards compatibility alias for the 'transform_graph'
        argument.
      instance_name: Optional unique instance name. Necessary iff multiple
        Trainer components are declared in the same pipeline.

    Raises:
      ValueError:
        - When both or neither of 'module_file' and user function
          (e.g., trainer_fn and run_fn) is supplied.
        - When both or neither of 'examples' and 'transformed_examples'
            is supplied.
        - When 'transformed_examples' is supplied but 'transform_graph'
            is not supplied.
    """
    if [bool(module_file), bool(run_fn), bool(trainer_fn)].count(True) != 1:
      raise ValueError(
          "Exactly one of 'module_file', 'trainer_fn', or 'run_fn' must be "
          "supplied.")

    if bool(examples) == bool(transformed_examples):
      raise ValueError(
          "Exactly one of 'example' or 'transformed_example' must be supplied.")

    if transform_output:
      absl.logging.warning(
          'The "transform_output" argument to the Trainer component has '
          'been renamed to "transform_graph" and is deprecated. Please update '
          "your usage as support for this argument will be removed soon.")
      transform_graph = transform_output
    if transformed_examples and not transform_graph:
      raise ValueError("If 'transformed_examples' is supplied, "
                       "'transform_graph' must be supplied too.")
    examples = examples or transformed_examples
    output = output or types.Channel(
        type=standard_artifacts.Model, artifacts=[standard_artifacts.Model()])
    model_run = model_run or types.Channel(
        type=standard_artifacts.ModelRun,
        artifacts=[standard_artifacts.ModelRun()])
    spec = TrainerSpec(
        examples=examples,
        transform_graph=transform_graph,
        schema=schema,
        base_model=base_model,
        hyperparameters=hyperparameters,
        train_args=train_args,
        eval_args=eval_args,
        module_file=module_file,
        run_fn=run_fn,
        trainer_fn=trainer_fn,
        custom_config=json_utils.dumps(custom_config),
        model=output,
        # TODO(b/158106209): change the model_run as optional output artifact
        model_run=model_run)
    super(Trainer, self).__init__(
        spec=spec,
        custom_executor_spec=custom_executor_spec,
        instance_name=instance_name)
