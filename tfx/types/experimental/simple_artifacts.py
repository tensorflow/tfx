# Lint as: python2, python3
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
"""A set of simple Artifact types for use with AI Platform Pipelines.

Experimental: the artifact definitions here are expected to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.types import artifact


class Dataset(artifact.Artifact):
  TYPE_NAME = 'Dataset'


class File(artifact.Artifact):
  TYPE_NAME = 'File'


class Statistics(artifact.Artifact):
  TYPE_NAME = 'Statistics'


class Model(artifact.Artifact):
  TYPE_NAME = 'Model'


class Metrics(artifact.Artifact):
  TYPE_NAME = 'Metrics'


class Schema(artifact.Artifact):
  TYPE_NAME = 'Schema'
