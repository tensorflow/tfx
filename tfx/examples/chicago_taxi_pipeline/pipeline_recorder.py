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

import argparse
import os
from distutils.dir_util import copy_tree

import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx.orchestration import metadata

print('TF version: {}'.format(tf.version.VERSION))
print('TFDV version: {}'.format(tfdv.version.__version__))
# Read artifact information from metadata store.
# import beam_dag_runner

def main(record_dir, pipeline_path, context_id):
  metadata_dir = os.path.join(os.environ['HOME'],
                              'tfx/tfx/examples/chicago_taxi_pipeline/',
                              'metadata.db')

  
  metadata_config = metadata.sqlite_metadata_connection_config(metadata_dir)
  with metadata.Metadata(metadata_config) as m:
    for artifact in m.store.get_artifacts()#_by_context(context_id):
      if artifact == 'ExternalArtifact':
        continue

      src_path = artifact.uri

      dest_path = src_path.replace(pipeline_path, "")
      dest_path = dest_path[:dest_path.rfind('/')] # remove trailing number
      dest_path = os.path.join(record_dir, dest_path)

      os.makedirs(dest_path, exist_ok=True)
      copy_tree(src_path, dest_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  pipeline_path = os.path.join(os.environ['HOME'],
                             "tfx/pipelines/chicago_taxi_beam/")
  record_dir = os.path.join(os.environ['HOME'],
                            'tfx/tfx/examples/chicago_taxi_pipeline/testdata')
  parser.add_argument(
      '--pipeline_path',
      type=str,
      default=pipeline_path,
      help='Path to pipeline')
  parser.add_argument(
      '--record_path',
      type=str,
      default=record_path,
      help='Path to record')
  parser.add_argument(
      '--context_id',
      type=str,
      default=None,
      help='Pipeline Context Id')
  # record_dir, pipeline_path = '', '', ''
  main(record_dir, pipeline_path, context_id)
