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
"""Recording pipeline from MLMD metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow_data_validation as tfdv
from tfx.orchestration import metadata

from distutils.dir_util import copy_tree
import os

print('TF version: {}'.format(tf.version.VERSION))
print('TFDV version: {}'.format(tfdv.version.__version__))
# Read artifact information from metadata store.
# import beam_dag_runner


metadata_dir = os.path.join(os.environ['HOME'],
                            'tfx/tfx/examples/chicago_taxi_pipeline/',
                            'metadata.db')
record_dir = os.path.join(os.environ['HOME'],
                          'tfx/tfx/examples/chicago_taxi_pipeline/testdata')

metadata_config = metadata.sqlite_metadata_connection_config(metadata_dir)
with metadata.Metadata(metadata_config) as m:
  for artifact in m.store.get_artifacts():
    if artifact == 'ExternalArtifact':
      continue
    src_path = artifact.uri
    pipeline_path = os.path.join(os.environ['HOME'],
                                 "tfx/pipelines/chicago_taxi_beam/")
    dest_path = src_path.replace(pipeline_path, "")
    dest_path = dest_path[:dest_path.rfind('/')] # remove trailing number
    dest_path = os.path.join(record_dir, dest_path)

    os.makedirs(dest_path, exist_ok=True)
    copy_tree(src_path, dest_path)
