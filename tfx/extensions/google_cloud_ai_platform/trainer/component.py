# Copyright 2021 Google LLC. All Rights Reserved.
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
"""TFX Trainer for training model on AI Platform."""
from typing import Any, Dict, Optional, Union

from tfx import types
from tfx.components.trainer import component as trainer_component
from tfx.dsl.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.trainer import executor
from tfx.orchestration import data_types
from tfx.proto import trainer_pb2


class Trainer(trainer_component.Trainer):
  """Cloud AI Platform Trainer component."""

  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.GenericExecutor)

  def __init__(self,
               examples: Optional[types.Channel] = None,
               transformed_examples: Optional[types.Channel] = None,
               transform_graph: Optional[types.Channel] = None,
               schema: Optional[types.Channel] = None,
               base_model: Optional[types.Channel] = None,
               hyperparameters: Optional[types.Channel] = None,
               module_file: Optional[Union[str,
                                           data_types.RuntimeParameter]] = None,
               run_fn: Optional[Union[str, data_types.RuntimeParameter]] = None,
               trainer_fn: Optional[Union[str,
                                          data_types.RuntimeParameter]] = None,
               train_args: Optional[Union[trainer_pb2.TrainArgs,
                                          data_types.RuntimeParameter]] = None,
               eval_args: Optional[Union[trainer_pb2.EvalArgs,
                                         data_types.RuntimeParameter]] = None,
               custom_config: Optional[Dict[str, Any]] = None):
    """Construct a Trainer component.

    Args:
      examples: A Channel of type `standard_artifacts.Examples`, serving as the
        source of examples used in training (required). May be raw or
        transformed.
      transformed_examples: Deprecated field. Please set `examples` instead.
      transform_graph: An optional Channel of type
        `standard_artifacts.TransformGraph`, serving as the input transform
        graph if present.
      schema:  An optional Channel of type `standard_artifacts.Schema`, serving
        as the schema of training and eval data. Schema is optional when 1)
        transform_graph is provided which contains schema. 2) user module
        bypasses the usage of schema, e.g., hardcoded.
      base_model: A Channel of type `Model`, containing model that will be used
        for training. This can be used for warmstart, transfer learning or model
        ensembling.
      hyperparameters: A Channel of type `standard_artifacts.HyperParameters`,
        serving as the hyperparameters for training module. Tuner's output best
        hyperparameters can be feed into this.
      module_file: A path to python module file containing UDF model definition.
        The module_file must implement a function named `run_fn` at its top
        level with function signature: `def
          run_fn(trainer.fn_args_utils.FnArgs)`, and the trained model must be
          saved to FnArgs.serving_model_dir when this function is executed.  For
          Estimator based Executor, The module_file must implement a function
          named `trainer_fn` at its top level. The function must have the
          following signature. def trainer_fn(trainer.fn_args_utils.FnArgs,
                         tensorflow_metadata.proto.v0.schema_pb2) -> Dict: ...
                           where the returned Dict has the following key-values.
            'estimator': an instance of tf.estimator.Estimator
            'train_spec': an instance of tf.estimator.TrainSpec
            'eval_spec': an instance of tf.estimator.EvalSpec
            'eval_input_receiver_fn': an instance of tfma EvalInputReceiver.
      run_fn:  A python path to UDF model definition function for generic
        trainer. See 'module_file' for details. Exactly one of 'module_file' or
        'run_fn' must be supplied if Trainer uses GenericExecutor (default).
      trainer_fn:  A python path to UDF model definition function for estimator
        based trainer. See 'module_file' for the required signature of the UDF.
        Exactly one of 'module_file' or 'trainer_fn' must be supplied if Trainer
        uses Estimator based Executor
      train_args: A proto.TrainArgs instance, containing args used for training
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is train on `train` split.
      eval_args: A proto.EvalArgs instance, containing args used for evaluation.
        Currently only splits and num_steps are available. Default behavior
        (when splits is empty) is evaluate on `eval` split.
      custom_config: A dict which contains addtional training job parameters
        that will be passed into user module.
    """
    super().__init__(
        examples=examples,
        transformed_examples=transformed_examples,
        transform_graph=transform_graph,
        schema=schema,
        base_model=base_model,
        hyperparameters=hyperparameters,
        train_args=train_args,
        eval_args=eval_args,
        module_file=module_file,
        run_fn=run_fn,
        trainer_fn=trainer_fn,
        custom_config=custom_config)
