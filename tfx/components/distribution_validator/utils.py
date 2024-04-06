# Copyright 2024 Google LLC. All Rights Reserved.
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
"""DistributionValidator utils."""

from tfx.proto import distribution_validator_pb2
from tfx.types import artifact
from tfx.utils import io_utils


def load_config_from_artifact(
    config_artifact: artifact.Artifact,
) -> distribution_validator_pb2.DistributionValidatorConfig:
  """Load a serialized DistributionValidatorConfig proto from artifact."""
  fpath = io_utils.get_only_uri_in_dir(config_artifact.uri)

  dv_config = distribution_validator_pb2.DistributionValidatorConfig()
  dv_config.ParseFromString(io_utils.read_bytes_file(fpath))
  return dv_config
