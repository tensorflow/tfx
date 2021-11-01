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

from typing import List, Optional

from absl import logging
from tfx import types
from tfx.components.example_validator import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class ExampleValidator(base_component.BaseComponent):
  """A TFX component to validate input examples.

  The ExampleValidator component uses [Tensorflow Data
  Validation](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv)
  to validate the statistics of some splits on input examples against a schema.

  The ExampleValidator component identifies anomalies in training and serving
  data. The component can be configured to detect different classes of anomalies
  in the data. It can:
    - perform validity checks by comparing data statistics against a schema that
      codifies expectations of the user.

  Schema Based Example Validation
  The ExampleValidator component identifies any anomalies in the example data by
  comparing data statistics computed by the StatisticsGen component against a
  schema. The schema codifies properties which the input data is expected to
  satisfy, and is provided and maintained by the user.

  ## Example
  ```
  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=infer_schema.outputs['schema'])
  ```

  Component `outputs` contains:
   - `anomalies`: Channel of type `standard_artifacts.ExampleAnomalies`.

  See [the ExampleValidator
  guide](https://www.tensorflow.org/tfx/guide/exampleval) for more details.
  """

  SPEC_CLASS = standard_component_specs.ExampleValidatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               statistics: types.Channel,
               schema: types.Channel,
               exclude_splits: Optional[List[str]] = None):
    """Construct an ExampleValidator component.

    Args:
      statistics: A Channel of type `standard_artifacts.ExampleStatistics`.
      schema: A Channel of type `standard_artifacts.Schema`. _required_
      exclude_splits: Names of splits that the example validator should not
        validate. Default behavior (when exclude_splits is set to None)
        is excluding no splits.
    """
    if exclude_splits is None:
      exclude_splits = []
      logging.info('Excluding no splits because exclude_splits is not set.')
    anomalies = types.Channel(type=standard_artifacts.ExampleAnomalies)
    spec = standard_component_specs.ExampleValidatorSpec(
        statistics=statistics,
        schema=schema,
        exclude_splits=json_utils.dumps(exclude_splits),
        anomalies=anomalies)
    super().__init__(spec=spec)
