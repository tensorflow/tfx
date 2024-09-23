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

"""Constants for [Evaluator](../../../guide/evaluator)."""

# Keys for artifact (custom) properties.
ARTIFACT_PROPERTY_BLESSED_KEY = 'blessed'
ARTIFACT_PROPERTY_CURRENT_MODEL_URI_KEY = 'current_model'
ARTIFACT_PROPERTY_CURRENT_MODEL_ID_KEY = 'current_model_id'
ARTIFACT_PROPERTY_BASELINE_MODEL_URI_KEY = 'baseline_model'
ARTIFACT_PROPERTY_BASELINE_MODEL_ID_KEY = 'baseline_model_id'
ARTIFACT_PROPERTY_BLESSING_MESSAGE_KEY = 'blessing_message'
ARTIFACT_PROPERTY_VALIDATION_RESULT_KEY = 'validation_result'


# Values for blessing results.
BLESSED_VALUE = 1
NOT_BLESSED_VALUE = 0

# File names for blessing results.
BLESSED_FILE_NAME = 'BLESSED'
NOT_BLESSED_FILE_NAME = 'NOT_BLESSED'

# File name for validations.tfrecord file produced by TFMA v2.
VALIDATIONS_TFRECORDS_FILE_NAME = 'validations.tfrecord'

# Key for evaluation results in executor output_dict.
EVALUATION_KEY = 'evaluation'

# Values for blessing messages.
RUBBER_STAMPED_AND_BLESSED_VALUE = (
    'The model was rubber stamped (no baseline models found) and blessed. '
    'Any change thresholds were ignored, but value thresholds were '
    'checked and passed.'
)
RUBBER_STAMPED_AND_NOT_BLESSED_VALUE = (
    'The model was rubber stamped (no baseline models found) and not blessed. '
    'Any change thresholds were ignored, but value thresholds were '
    'checked and failed.'
)
NOT_RUBBER_STAMPED_AND_NOT_BLESSED_VALUE = (
    'The model was not rubber stamped (a baseline model was found) and not '
    'blessed. Change thresholds and value thresholds were checked and there '
    'were failures.'
)


def get_no_validation_file_value(validation_path: str) -> str:
  return (
      f'No validations.tfrecords file found at {validation_path}. The '
      '"blessed" custom_property will not be set.'
  )


# Spanner has a 2.5MB limit on string values, see go/spanner-limits.
# kMaxStringSizeCharacters is not available in Python, so we copy its
# calculation here.
VALIDATION_RESULT_MAX_CHARACTERS = int((10 << 20) / 4)  # 2,621,440 characters
