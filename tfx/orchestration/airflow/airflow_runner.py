# Lint as: python2, python3
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
"""Deprecated definition of Airflow TFX runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import
from tfx.orchestration.airflow import airflow_dag_runner

AirflowDAGRunner = deprecation.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='airflow_runner.AirflowDAGRunner',
    name='airflow_dag_runner.AirflowDagRunner',
    func_or_class=airflow_dag_runner.AirflowDagRunner)
