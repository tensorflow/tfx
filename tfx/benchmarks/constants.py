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
"""Constants used for beam pipeline execution and tfma, tfmav_2, tft tests. """

DEFAULT_MODE = "default"
LOCAL_SCALED_EXECUTION_MODE = "local_scaled_execution"
CLOUD_DATAFLOW_MODE = "cloud_dataflow"
FLINK_ON_K8S_MODE = "flink_on_k8s"
modes = [DEFAULT_MODE, LOCAL_SCALED_EXECUTION_MODE, CLOUD_DATAFLOW_MODE,
         FLINK_ON_K8S_MODE]


SCHEMA_PATH = "../examples/chicago_taxi_pipeline/data/user_provided_schema/" \
              "schema.pbtxt"
