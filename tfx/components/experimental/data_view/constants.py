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
"""Defines public constants."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Custom properties attached to the Examples artifacts output by DataViewBinder.
# Components can use these properties to construct the right TFXIO to access
# the data through the DataView.
DATA_VIEW_URI_PROPERTY_KEY = 'data_view_uri'
DATA_VIEW_CREATE_TIME_KEY = 'data_view_create_time_since_epoch'
