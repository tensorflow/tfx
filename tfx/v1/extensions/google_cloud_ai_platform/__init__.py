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
"""Google cloud AI platform module."""

from tfx.extensions.google_cloud_ai_platform.bulk_inferrer.component import CloudAIBulkInferrerComponent as BulkInferrer
from tfx.extensions.google_cloud_ai_platform.pusher.component import Pusher
from tfx.extensions.google_cloud_ai_platform.trainer.component import Trainer
from tfx.extensions.google_cloud_ai_platform.trainer.executor import ENABLE_UCAIP_KEY
from tfx.extensions.google_cloud_ai_platform.trainer.executor import JOB_ID_KEY
from tfx.extensions.google_cloud_ai_platform.trainer.executor import TRAINING_ARGS_KEY
from tfx.extensions.google_cloud_ai_platform.trainer.executor import UCAIP_REGION_KEY
from tfx.extensions.google_cloud_ai_platform.tuner.component import Tuner
from tfx.v1.extensions.google_cloud_ai_platform import experimental
