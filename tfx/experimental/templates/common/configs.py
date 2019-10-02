# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static pipeline configurations."""

from tfx.proto import example_gen_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

# Developer TODO: Adjust configurations with own data and model specification.

PIPELINE_NAME = "skeleton-pipeline-template"

PIPELINE_ROOT = "./"

METADATA_ROOT = "./"

_query_sample_rate = 0.1
INPUT_CONFIG = example_gen_pb2.Input(
    splits=[
        example_gen_pb2.Input.Split(
            name="single_split",
            pattern="""
                SELECT fare, payment_type, tips
                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                LIMIT (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF
                    < {query_sample_rate}
            """.format(query_sample_rate=_query_sample_rate))
    ],)

PREPROCESSING_FN = "transform.preprocessing.preprocessing_fn"

TRAINER_FN = "transform.model.trainer_fn"
TRAIN_ARGS = trainer_pb2.TrainArgs(num_steps=10000)
EVAL_ARGS = trainer_pb2.EvalArgs(num_steps=10000)

PUSH_DESTINATION = pusher_pb2.PushDestination(
    filesystem=pusher_pb2.PushDestination.Filesystem(base_directory="./"))

ENABLE_CACHE = True

ADDITIONAL_PIPELINE_ARGS = {}
