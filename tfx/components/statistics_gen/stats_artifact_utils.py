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
"""StatisticsGen specific artifact utilities."""

import os

import tensorflow_data_validation as tfdv
from tfx.types import artifact
from tfx.types import artifact_utils


BINARY_PB_BASENAME = 'FeatureStats.pb'
TFRECORD_BASENAME = 'stats_tfrecord'
SHARDED_STATS_PREFIX = 'FeatureStats'


def load_statistics(stats_artifact: artifact.Artifact,
                    split: str) -> tfdv.DatasetListView:
  stats_dir = artifact_utils.get_split_uri([stats_artifact], split)
  if artifact_utils.is_artifact_version_older_than(
      stats_artifact, artifact_utils._ARTIFACT_VERSION_FOR_STATS_UPDATE):  # pylint: disable=protected-access
    stats = tfdv.load_statistics(os.path.join(stats_dir, TFRECORD_BASENAME))
  else:
    stats = tfdv.load_stats_binary(os.path.join(stats_dir, BINARY_PB_BASENAME))
  return tfdv.DatasetListView(stats)
