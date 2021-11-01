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
EXAMPLES_DATA_FORMAT_LABEL = 'examples_data_format'
SCHEMA_PATH_LABEL = 'schema_path'
ANALYZE_DATA_PATHS_LABEL = 'analyze_data_paths'
ANALYZE_PATHS_FILE_FORMATS_LABEL = 'analyze_paths_file_formats'
TRANSFORM_DATA_PATHS_LABEL = 'transform_data_paths'
TRANSFORM_PATHS_FILE_FORMATS_LABEL = 'transform_paths_file_formats'
DISABLE_STATISTICS_LABEL = 'disable_statistics'
MODULE_FILE = 'module_file'
MODULE_PATH = 'module_path'
PREPROCESSING_FN = 'preprocessing_fn'
STATS_OPTIONS_UPDATER_FN = 'stats_options_updater_fn'
CUSTOM_CONFIG = 'custom_config'
BEAM_PIPELINE_ARGS = 'pipeline_args'
DATA_VIEW_LABEL = 'data_view_uri'
# This label is currently not used externally.
EXAMPLES_METADATA_LABEL = 'examples_metadata'
CACHE_INPUT_PATH_LABEL = 'cache_input_path'
# This label is used to determine whether Transform should execute in
# `tf.compat.v1` mode or not.
FORCE_TF_COMPAT_V1_LABEL = 'force_tf_compat_v1'

# Output labels.
# TODO(b/72214804): Ideally per-set stats and materialization output paths
# should be output labels, but they require multiple values. Change this if/when
# we can add multiple outputs to a single processor label.
PER_SET_STATS_OUTPUT_PATHS_LABEL = 'per_set_stats_output_paths'
TRANSFORM_MATERIALIZE_OUTPUT_PATHS_LABEL = (
    'transform_materialize_output_paths')
TRANSFORM_METADATA_OUTPUT_PATH_LABEL = 'transform_output_path'
CACHE_OUTPUT_PATH_LABEL = 'cache_output_path'
TEMP_OUTPUT_LABEL = 'temp_path'
PRE_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL = 'pre_transform_output_schema_path'
PRE_TRANSFORM_OUTPUT_STATS_PATH_LABEL = 'pre_transform_output_stats_path'
POST_TRANSFORM_OUTPUT_SCHEMA_PATH_LABEL = 'post_transform_output_schema_path'
POST_TRANSFORM_OUTPUT_STATS_PATH_LABEL = 'post_transform_output_stats_path'
POST_TRANSFORM_OUTPUT_ANOMALIES_PATH_LABEL = (
    'post_transform_output_anomalies_path')

# Examples File Format
FORMAT_TFRECORD = 'tfrecords_gzip'
