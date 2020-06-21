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
"""In process component launcher which launches python executors in process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, List, Optional, Text, Type
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma

import absl
from six import with_metaclass

from tfx import types
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.components.base import base_executor
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import publisher
from tfx.orchestration.config import base_component_config
from tfx.orchestration.launcher import in_process_component_launcher
from tfx.components.transform.executor import TRANSFORM_GRAPH_KEY, TRANSFORMED_EXAMPLES_KEY#EXAMPLES_KEY, SCHEMA_KEY
from tfx.experimental.dummy_executor import BaseDummyExecutor
from tfx.components.trainer import fn_args_utils
from tfx.components.evaluator import constants

from tfx.components.util import udf_utils
from tfx.types import artifact_utils
from tfx.utils import path_utils


from tfx.components.trainer.executor import Executor as TrainerExecutor

class MyDummyComponentLauncher(in_process_component_launcher.InProcessComponentLauncher):
  """Responsible for launching a dummy executor.

  The executor will be launched in the same process of the rest of the
  component, i.e. its driver and publisher.
  """
  def __init__(self,
                component: base_node.BaseNode,
                pipeline_info: data_types.PipelineInfo,
                driver_args: data_types.DriverArgs,
                metadata_connection: metadata.Metadata,
                beam_pipeline_args: List[Text],
                additional_pipeline_args: Dict[Text, Any],
                component_config: Optional[
                    base_component_config.BaseComponentConfig] = None
            ):
    absl.logging.info("Launching MyDummyComponentLauncher")
    super(MyDummyComponentLauncher, self).__init__(component,
                                                   pipeline_info,
                                                   driver_args,
                                                   metadata_connection,
                                                   beam_pipeline_args,
                                                   additional_pipeline_args,
                                                   component_config)
    self._record_dir = os.path.join(os.environ['HOME'], 'tfx/tfx/examples/chicago_taxi_pipeline/testdata')
    self.dummy_dict = {}    # component_id: dummy_executor 
    for component_id in ['CsvExampleGen', 'StatisticsGen', 'SchemaGen', 'ExampleValidator', 'Evaluator', 'Pusher']:
    # for component_id in ['CsvExampleGen', 'StatisticsGen', 'SchemaGen', 'ExampleValidator', 'Transform', 'Trainer', 'Evaluator', 'Pusher']:
      self.dummy_dict[component_id] = BaseDummyExecutor


  def _run_executor(self, execution_id: int,
                    input_dict: Dict[Text, List[types.Artifact]],
                    output_dict: Dict[Text, List[types.Artifact]],
                    exec_properties: Dict[Text, Any]) -> None:
    """Execute underlying component implementation."""
    component_id = self._component_info.component_id
    if component_id not in self.dummy_dict.keys():
      super(MyDummyComponentLauncher, self)._run_executor(execution_id, 
                                                          input_dict, 
                                                          output_dict,
                                                          exec_properties)
      if component_id == 'Transform':
        '''transform_graph metadata/schema.pbtxt 
          transform_fn/assets
              saved_model.pb  
              variables
              transformed_metadata/schema.pbtxt
        TRANSFORMED_EXAMPLES_KEY transformed_examples/ eval  train. *.gz
        '''
        # TODO: may need verifier for input
        # TODO: verifier for output
        transform_graph_uri = artifact_utils.get_single_uri(output_dict[TRANSFORM_GRAPH_KEY]) 
        expected_transform_graph_uri = transform_graph_uri.replace(os.path.join(os.environ['HOME'], "tfx/pipelines/chicago_taxi_beam/"), "")
        expected_transform_graph_uri = expected_transform_graph_uri[:expected_transform_graph_uri.rfind('/')] # remove trailing number
        expected_transform_graph_uri = os.path.join(self._record_dir, expected_transform_graph_uri)
        transform_output = tft.TFTransformOutput(transform_graph_uri)
        expected_transform_output = tft.TFTransformOutput(expected_transform_graph_uri)
        print("expected_transform_output.transformed_metadata ", expected_transform_output.transformed_metadata )
        print("transform_output.transformed_metadata", transform_output.transformed_metadata)
        assert expected_transform_output.transformed_metadata == transform_output.transformed_metadata
      elif component_id == 'Trainer':
        # Assuming using a same model
        model_uri = artifact_utils.get_single_uri(output_dict[constants.MODEL_KEY])
        model_path = path_utils.eval_model_path(model_uri) #/Users/sujipark/tfx/pipelines/chicago_taxi_beam/Trainer/model/173/serving_model_dir
        #artifact_utils.get_single_uri(output_dict[])
        # model = tfma.default_eval_shared_model(eval_saved_model_path=model_path)
        print("model_path", model_path)
        saved_model = tf.saved_model.load(model_path)
        structured_outputs = saved_model.structured_outputs
        structured_input_signature = saved_model.structured_input_signature
        expected_model_uri = model_uri.replace(os.path.join(os.environ['HOME'], "tfx/pipelines/chicago_taxi_beam/"), "")
        expected_model_uri = os.path.join(self._record_dir, expected_model_uri[:expected_model_uri.rfind('/')])
        expected_model = tf.saved_model.load(path_utils.eval_model_path(expected_model_uri))
        expected_structured_outputs = expected_model.structured_outputs
        expected_structured_input_signature = expected_model.structured_input_signature
        print("expected_structured_outputs", expected_structured_outputs)
        print("expected_structured_input_signature", expected_structured_input_signature)
        assert expected_structured_outputs == structured_outputs
        assert expected_structured_input_signature == structured_input_signature

    else:
      executor_context = base_executor.BaseExecutor.Context(
          beam_pipeline_args=self._beam_pipeline_args,
          tmp_dir=os.path.join(self._pipeline_info.pipeline_root, '.temp', ''),
          unique_id=str(execution_id))
      executor = self.dummy_dict[component_id](component_id, self._record_dir, executor_context)
      absl.logging.info("Running executor [%s]", executor)
      executor.Do(input_dict, output_dict, exec_properties)
