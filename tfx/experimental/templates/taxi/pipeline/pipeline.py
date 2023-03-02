# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TFX taxi template pipeline definition.

This file defines TFX pipeline and various components in the pipeline.
"""

from typing import Any, Dict, List, Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx

from ml_metadata.proto import metadata_store_pb2

@tfx.dsl.components.component
def CustomComp(a: tfx.dsl.components.Parameter[List[str]],
               b: tfx.dsl.components.Parameter[Dict[str, int]]):
  print(f'#### a: {a}, type(a) : {type(a)}')
  print(f'#### b`: {b}, type(b) : {type(b)}')


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    # TODO(step 7): (Optional) Uncomment here to use BigQuery as a data source.
    # query: str,
    preprocessing_fn: str,
    run_fn: str,
    train_args: tfx.proto.TrainArgs,
    eval_args: tfx.proto.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: str,
    schema_path: Optional[str] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[str]] = None,
    ai_platform_training_args: Optional[Dict[str, str]] = None,
    ai_platform_serving_args: Optional[Dict[str, Any]] = None,
) -> tfx.dsl.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  c = CustomComp(a=['foo', 'bar'], b={'foo': 1, 'bar': 2})
  components = [c]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Change this value to control caching of execution results. Default value
      # is `False`.
      # enable_cache=True,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )
