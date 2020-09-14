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
"""Base class for TFX resolvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Dict, Text

from six import with_metaclass

from tfx import types
from tfx.dsl.resolvers import resolver_result
from tfx.orchestration import data_types
from tfx.orchestration import metadata


class BaseResolver(with_metaclass(abc.ABCMeta, object)):
  """Base class for resolver.

  Resolver is the logical unit that will be used optionally for input selection.
  A resolver subclass must override the resolve() function which takes a
  read-only MLMD handler and a dict of <key, Channel> as parameters and produces
  a ResolveResult instance.
  """

  @abc.abstractmethod
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> resolver_result.ResolveResult:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      pipeline_info: PipelineInfo of the current pipeline. We do not want to
        query artifacts across pipeline boundary.
      metadata_handler: a read-only handler to query MLMD.
      source_channels: a key -> channel dict which contains the info of the
        source channels.

    Returns:
      a ResolveResult instance.

    """
    raise NotImplementedError
