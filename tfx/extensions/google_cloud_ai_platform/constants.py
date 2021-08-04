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
"""Google Cloud AI Platform constants module."""
from tfx.utils import doc_controls

ENABLE_VERTEX_KEY = doc_controls.documented(
    obj='ai_platform_enable_vertex',
    doc='Keys to the items in custom_config of Trainer and Pusher for enabling'
    ' Vertex AI.')

VERTEX_REGION_KEY = doc_controls.documented(
    obj='ai_platform_vertex_region',
    doc='Keys to the items in custom_config of Trainer and Pusher for '
    'specifying the region of Vertex AI.')

# Prediction container registry: https://gcr.io/cloud-aiplatform/prediction.
VERTEX_CONTAINER_IMAGE_URI_KEY = doc_controls.documented(
    obj='ai_platform_vertex_container_image_uri',
    doc='Keys to the items in custom_config of Pusher/BulkInferrer for the '
    'serving container image URI in Vertex AI.')

# Keys to the items in custom_config passed as a part of exec_properties.
SERVING_ARGS_KEY = doc_controls.documented(
    obj='ai_platform_serving_args',
    doc='Keys to the items in custom_config of Pusher/BulkInferrer for passing '
    'serving args to AI Platform.')

ENDPOINT_ARGS_KEY = doc_controls.documented(
    obj='endpoint',
    doc='Keys to the items in custom_config of Pusher/BulkInferrer for optional'
    ' endpoint override (CAIP).')
