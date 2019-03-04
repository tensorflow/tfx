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
"""Labels recognized by the transform executor."""

# Input labels.
EXAMPLES_FILE_FORMAT_LABEL = 'examples_file_format'
EXAMPLES_DATA_FORMAT_LABEL = 'examples_data_format'
SCHEMA_PATH_LABEL = 'schema_path'
ANALYZE_AND_TRANSFORM_DATA_PATHS_LABEL = ('analyze_and_transform_data_paths')
TRANSFORM_ONLY_DATA_PATHS_LABEL = 'transform_only_data_paths'
COMPUTE_STATISTICS_LABEL = 'compute_statistics'
PREPROCESSING_FN = 'preprocessing_fn'
TFT_STATISTICS_USE_TFDV_LABEL = 'tft_statistics_use_tfdv'
BEAM_PIPELINE_ARGS = 'pipeline_args'
# This label is currently not used externally.
EXAMPLES_METADATA_LABEL = 'examples_metadata'

# Output labels.
# TODO(b/72214804): Ideally per-set stats and materialization output paths
# should be output labels, but they require multiple values. Change this if/when
# we can add multiple outputs to a single processor label.
PER_SET_STATS_OUTPUT_PATHS_LABEL = 'per_set_stats_output_paths'
TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL = (
    'transform_materialize_output_paths')
TRANSFORM_METADATA_OUTPUT_PATH_LABEL = 'transform_output_path'
TEMP_OUTPUT_LABEL = 'temp_path'

# Transform Data Format
# Indicates that the data format is tf.Example.
FORMAT_TF_EXAMPLE = 'FORMAT_TF_EXAMPLE'
# Indicates that the data format is tf.SequenceExample.
FORMAT_TF_SEQUENCE_EXAMPLE = 'FORMAT_TF_SEQUENCE_EXAMPLE'
# Indicates the data format is a custom proto.
FORMAT_PROTO = 'FORMAT_PROTO'
