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
"""TFX ExampleValidator component definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional, Text, Union

from absl import logging
from tfx import types
from tfx.components.schema_gen import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.types import standard_artifacts
from tfx.types.standard_component_specs import SchemaGenSpec
from tfx.utils import json_utils


class SchemaGen(base_component.BaseComponent):
  """A TFX SchemaGen component to generate a schema from the training data.

  The SchemaGen component uses [TensorFlow Data
  Validation](https://www.tensorflow.org/tfx/data_validation) to
  generate a schema from input statistics.  The following TFX libraries use the
  schema:
    - TensorFlow Data Validation
    - TensorFlow Transform
    - TensorFlow Model Analysis

  In a typical TFX pipeline, the SchemaGen component generates a schema which is
  is consumed by the other pipeline components.

  Please see https://www.tensorflow.org/tfx/data_validation for more details.

  ## Example
  ```
    # Generates schema based on statistics files.
    infer_schema = SchemaGen(statistics=statistics_gen.outputs['statistics'])
  ```

  Component `outputs` contains:
   - `schema`: Channel of type `standard_artifacts.Schema` for schema result.
  """
  # TODO(b/123941608): Update pydoc about how to use a user provided schema

  SPEC_CLASS = SchemaGenSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(
      self,
      statistics: Optional[types.Channel] = None,
      infer_feature_shape: Optional[Union[bool,
                                          data_types.RuntimeParameter]] = True,
      exclude_splits: Optional[List[Text]] = None):
    """Constructs a SchemaGen component.

    Args:
      statistics: A Channel of `ExampleStatistics` type (required if spec is not
        passed). This should contain at least a `train` split. Other splits are
        currently ignored. _required_
      infer_feature_shape: Boolean (or RuntimeParameter) value indicating
        whether or not to infer the shape of features. If the feature shape is
        not inferred, downstream Tensorflow Transform component using the schema
        will parse input as tf.SparseTensor. Default to True if not set.
      exclude_splits: Names of splits that will not be taken into consideration
        when auto-generating a schema. Default behavior (when exclude_splits is
        set to None) is excluding no splits.
    """
    if exclude_splits is None:
      exclude_splits = []
      logging.info('Excluding no splits because exclude_splits is not set.')
    schema = types.Channel(type=standard_artifacts.Schema)
    if isinstance(infer_feature_shape, bool):
      infer_feature_shape = int(infer_feature_shape)
    spec = SchemaGenSpec(
        statistics=statistics,
        infer_feature_shape=infer_feature_shape,
        exclude_splits=json_utils.dumps(exclude_splits),
        schema=schema)
    super(SchemaGen, self).__init__(spec=spec)
