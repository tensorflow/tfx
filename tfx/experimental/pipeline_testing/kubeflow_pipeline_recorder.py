# Lint as: python3
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
"""Recording pipeline using Kubeflow gRPC metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app
from absl import flags
from ml_metadata.proto import metadata_store_pb2

from tfx.experimental.pipeline_testing import pipeline_recorder_utils


FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Path to record the pipeline outputs.')
flags.DEFINE_integer('local_port', None, 'Local port number.')
flags.DEFINE_string('run_id', None, 'KFP Pipeline Run Id')

flags.mark_flag_as_required('output_dir')
flags.mark_flag_as_required('local_port')
flags.mark_flag_as_required('run_id')

def main(unused_argv):
  connection_config = metadata_store_pb2.MetadataStoreClientConfig()
  connection_config.host = 'localhost'
  connection_config.port = FLAGS.local_port
  pipeline_recorder_utils.record_kfp_pipeline(connection_config,
                                              FLAGS.run_id,
                                              FLAGS.output_dir)

if __name__ == '__main__':
  app.run(main)
