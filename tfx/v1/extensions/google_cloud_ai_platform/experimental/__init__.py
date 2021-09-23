# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Types used in Google Cloud AI Platform under experimental stage."""

from tfx.extensions.google_cloud_ai_platform.bulk_inferrer.executor import SERVING_ARGS_KEY as BULK_INFERRER_SERVING_ARGS_KEY
from tfx.extensions.google_cloud_ai_platform.constants import ENDPOINT_ARGS_KEY
# PUSHER_SERVING_ARGS_KEY is deprecated.
# Please use tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY instead.
from tfx.extensions.google_cloud_ai_platform.constants import SERVING_ARGS_KEY as PUSHER_SERVING_ARGS_KEY
from tfx.extensions.google_cloud_ai_platform.tuner.executor import REMOTE_TRIALS_WORKING_DIR_KEY
from tfx.extensions.google_cloud_ai_platform.tuner.executor import TUNING_ARGS_KEY
