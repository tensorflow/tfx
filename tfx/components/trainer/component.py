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

from typing import Any, Dict, Optional, Union

from absl import logging
from tfx import types
from tfx.components.trainer import executor
from tfx.components.util import udf_utils
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.proto import trainer_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class Trainer(base_component.BaseComponent):
  """A TFX component to train a TensorFlow model.

  The Trainer component is used to train and eval a model using given inputs and
  a user-supplied run_fn function.

  An example of `run_fn()` can be found in the [user-supplied
  code](https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py)
  of the TFX penguin pipeline example.

  *Note:* This component trains locally. For cloud distributed training, please
  refer to [Cloud AI Platform
  Trainer](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_ai_platform/trainer).

  ## Example
  ```
  # Uses user-provided Python function that trains a model using TF.
  trainer = Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      schema=infer_schema.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=proto.TrainArgs(splits=['train'], num_steps=10000),
      eval_args=proto.EvalArgs(splits=['eval'], num_steps=5000))
  ```

  Component `outputs` contains:
   - `model`: Channel of type `standard_artifacts.Model` for trained model.
   - `model_run`: Channel of type `standard_artifacts.ModelRun`, as the working
                  dir of models, can be used to output non-model related output
                  (e.g., TensorBoard logs).

  Please see [the Trainer guide](https://www.tensorflow.org/tfx/guide/trainer)
  for more details.
  """

  SPEC_CLASS = standard_component_specs.TrainerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.GenericExecutor)

  def __init__(
      self,
      examples: Optional[types.Channel] = None,
      transformed_examples: Optional[types.Channel] = None,
      transform_graph: Optional[types.Channel] = None,
      schema: Optional[types.Channel] = None,
      base_model: Optional[types.Channel] = None,
      hyperparameters: Optional[types.Channel] = None,
      module_file: Optional[Union[str, data_types.RuntimeParameter]] = None,
      run_fn: Optional[Union[str, data_types.RuntimeParameter]] = None,
      # TODO(b/147702778): deprecate trainer_fn.
      trainer_fn: Optional[Union[str, data_types.RuntimeParameter]] = None,
      train_args: Optional[Union[trainer_pb2.TrainArgs,
                                 data_types.RuntimeParameter]] = None,
      eval_args: Optional[Union[trainer_pb2.EvalArgs,
                                data_types.RuntimeParameter]] = None,
      custom_config: Optional[Union[Dict[str, Any],
                                    data_types.RuntimeParameter]] = None,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
    """Construct a Trainer component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as
        the source of examples used in training (required). May be raw or
        transformed.
      transformed_examples: Deprecated (no compatibility guarantee). Please set
        'examples' instead.
      transform_graph: An optional Channel of type
        `standard_artifacts.TransformGraph`, serving as the input transform
        graph if present.
      schema:  An optional Channel of type `standard_artifacts.Schema`, serving
        as the schema of training and eval data. Schema is optional when
        1) transform_graph is provided which contains schema.
        2) user module bypasses the usage of schema, e.g., hardcoded.
      base_model: A Channel of type `Model`, containing model that will be used
        for training. This can be used for warmstart, transfer learning or
        model ensembling.
      hyperparameters: A Channel of type `standard_artifacts.HyperParameters`,
        serving as the hyperparameters for training module. Tuner's output best
        hyperparameters can be feed into this.
      module_file: A path to python module file containing UDF model definition.
        The module_file must implement a function named `run_fn` at its top
        level with function signature:
          `def run_fn(trainer.fn_args_utils.FnArgs)`,
        and the trained model must be saved to FnArgs.serving_model_dir when
        this function is executed.

        For Estimator based Executor, The module_file must implement a function
        named `trainer_fn` at its top level. The function must have the
        following signature.
          def trainer_fn(trainer.fn_args_utils.FnArgs,
                         tensorflow_metadata.proto.v0.schema_pb2) -> Dict:
            ...
          where the returned Dict has the following key-values.
            'estimator': an instance of tf.estimator.Estimator
            'train_spec': an instance of tf.estimator.TrainSpec
            'eval_spec': an instance of tf.estimator.EvalSpec
            'eval_input_receiver_fn': an instance of tfma EvalInputReceiver.
        Exactly one of 'module_file' or 'run_fn' must be supplied if Trainer
        uses GenericExecutor (default). Use of a RuntimeParameter for this
        argument is experimental.
      run_fn:  A python path to UDF model definition function for generic
        trainer. See 'module_file' for details. Exactly one of 'module_file' or
        'run_fn' must be supplied if Trainer uses GenericExecutor (default).
         Use of a RuntimeParameter for this argument is experimental.
      trainer_fn:  A python path to UDF model definition function for estimator
        based trainer. See 'module_file' for the required signature of the UDF.
        Exactly one of 'module_file' or 'trainer_fn' must be supplied if Trainer
        uses Estimator based Executor. Use of a RuntimeParameter for this
        argument is experimental.
      train_args: A proto.TrainArgs instance, containing args used for training
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is train on `train` split.
      eval_args: A proto.EvalArgs instance, containing args used for evaluation.
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is evaluate on `eval` split.
      custom_config: A dict which contains addtional training job parameters
        that will be passed into user module.
      custom_executor_spec: Optional custom executor spec. Deprecated (no
        compatibility guarantee), please customize component directly.

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

    if transformed_examples and not transform_graph:
      raise ValueError("If 'transformed_examples' is supplied, "
                       "'transform_graph' must be supplied too.")

    if custom_executor_spec:
      logging.warning(
          "`custom_executor_spec` is deprecated. Please customize component directly."
      )
    if transformed_examples:
      logging.warning(
          "`transformed_examples` is deprecated. Please use `examples` instead."
      )
    examples = examples or transformed_examples
    model = types.Channel(type=standard_artifacts.Model)
    model_run = types.Channel(type=standard_artifacts.ModelRun)
    spec = standard_component_specs.TrainerSpec(
        examples=examples,
        transform_graph=transform_graph,
        schema=schema,
        base_model=base_model,
        hyperparameters=hyperparameters,
        train_args=train_args or trainer_pb2.TrainArgs(),
        eval_args=eval_args or trainer_pb2.EvalArgs(),
        module_file=module_file,
        run_fn=run_fn,
        trainer_fn=trainer_fn,
        custom_config=(custom_config
                       if isinstance(custom_config, data_types.RuntimeParameter)
                       else json_utils.dumps(custom_config)),
        model=model,
        model_run=model_run)
    super().__init__(spec=spec, custom_executor_spec=custom_executor_spec)

    if udf_utils.should_package_user_modules():
      # In this case, the `MODULE_PATH_KEY` execution property will be injected
      # as a reference to the given user module file after packaging, at which
      # point the `MODULE_FILE_KEY` execution property will be removed.
      udf_utils.add_user_module_dependency(
          self, standard_component_specs.MODULE_FILE_KEY,
          standard_component_specs.MODULE_PATH_KEY)
