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
"""Recording pipeline from MLMD metadata."""

from absl import app
from absl import flags

from tfx.experimental.pipeline_testing import pipeline_recorder_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Path to record the pipeline outputs.')
# metadata_db_uri is required for recording beam pipeline.
flags.DEFINE_string('metadata_db_uri', None, 'Path to metadata db.')
# host and port are required for recording KFP pipeline.
flags.DEFINE_string('host', None, 'Hostname of the metadata grpc server.')
flags.DEFINE_integer('port', None, 'Port number of the metadata grpc server.')
# If run_id is not specified, pipeline_name must be specified.
flags.DEFINE_string('pipeline_name', None, 'Name of the pipeline.')
flags.DEFINE_string(
    'run_id', None, 'Pipeline Run Id'
    '(default=latest run, must specify pipeline_name).')

flags.mark_flag_as_required('output_dir')


def main(unused_argv):
  pipeline_recorder_utils.record_pipeline(FLAGS.output_dir,
                                          FLAGS.metadata_db_uri, FLAGS.host,
                                          FLAGS.port, FLAGS.pipeline_name,
                                          FLAGS.run_id)


if __name__ == '__main__':
  app.run(main)
