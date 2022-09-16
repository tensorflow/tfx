# Copyright 2022 Google LLC. All Rights Reserved.
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
"""TFX DistributionValidator component definition."""

from typing import List, Optional, Tuple

from tfx import types
from tfx.components.distribution_validator import executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.proto import distribution_validator_pb2
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils


class DistributionValidator(base_component.BaseComponent):
  """TFX DistributionValidator component.

  Identifies distribution shifts between datasets by examining their summary
  statistics.
  """
  SPEC_CLASS = standard_component_specs.DistributionValidatorSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               statistics: types.BaseChannel,
               baseline_statistics: types.BaseChannel,
               config: distribution_validator_pb2.DistributionValidatorConfig,
               include_split_pairs: Optional[List[Tuple[str, str]]] = None):
    """Construct a DistributionValidation component.

    Args:
      statistics: A BaseChannel of type `standard_artifacts.ExampleStatistics`.
      baseline_statistics: A BaseChannel of type
        `standard_artifacts.ExampleStatistics` to which the distribution from
        `statistics` will be compared.
      config: A DistributionValidationConfig that defines configuration for the
        DistributionValidator.
      include_split_pairs: Pairs of split names that DistributionValidator
        should be run on. Default behavior if not supplied is to run on pairs of
        the same splits (i.e., (train, train), (test, test), etc.).
        Order is (statistics, baseline_statistics).
    """
    anomalies = types.Channel(type=standard_artifacts.ExampleAnomalies)
    spec = standard_component_specs.DistributionValidatorSpec(
        **{
            standard_component_specs.STATISTICS_KEY:
                statistics,
            standard_component_specs.BASELINE_STATISTICS_KEY:
                baseline_statistics,
            standard_component_specs.DISTRIBUTION_VALIDATOR_CONFIG_KEY: config,
            standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
                json_utils.dumps(include_split_pairs),
            standard_component_specs.ANOMALIES_KEY:
                anomalies
        })
    super().__init__(spec=spec)
