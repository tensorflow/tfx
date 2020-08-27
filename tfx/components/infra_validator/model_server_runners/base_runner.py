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
"""Module for shared interface of every model server runners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Text

import six


class BaseModelServerRunner(six.with_metaclass(abc.ABCMeta, object)):
  """Shared interface of all model server runners.

  Model server runner is responsible for managing the model server job and
  relevant resources in the serving platform. For example, model server runner
  for kubernetes will launch a Pod of model server with required resources
  allocated, and tear down all the kubernetes resources once infra validation
  is done. Note that model server runner does *not* interact with model server
  app.

  Model server job have 5 states: Initial, Scheduled, Running, Aborted, and End.
  Each state transition is depicted in the diagram below.

  ```
             +-----------+
             |  Initial  |
             +-----+-----+
                   | Start()
             +-----v-----+
          +--+ Scheduled |
          |  +-----+-----+
          |        | WaitUntilRunning()
          |  +-----v-----+
          +--+  Running  |
          |  +-----+-----+
          |        |
    +-----v-----+  |
    |  Aborted  +--+ Stop()
    +-----------+  |
                   |
             +-----v-----+
             |    End    |
             +-----------+
  ```

  At any step, the job can be aborted in the serving platform. Model server
  runner will NOT recover a job from failure (even if it can) and regard the
  abortion as a validation failure.

  All the infra validation logic (waiting for model loaded, sending requests,
  measuring metrics, etc.) will happen when model server job has reached Running
  state. This is not a scope of model server runner work.

  Depending on the serving platform, some of the states might be the same. For
  example, in a GCP cloud AI prediction service we have a global model server
  instance running, which makes Scheduled state and Running state
  indistinguishable. In such case, `WaitUntilRunning()` action will be a no-op.
  """

  @abc.abstractmethod
  def __repr__(self) -> Text:
    pass

  @abc.abstractmethod
  def GetEndpoint(self) -> Text:
    """Get an endpoint to the model server to connect to.

    Endpoint will be available after the model server job has reached the
    Running state.

    Raises:
      AssertionError: if runner hasn't reached the Running state.
    """

  @abc.abstractmethod
  def Start(self) -> None:
    """Start the model server in non-blocking manner.

    `Start()` will transition the job state from Initial to Scheduled. Serving
    platform will turn the job into Running state in the future.

    In `Start()`, model server runner should prepare the resources model server
    requires including config files, environment variables, volumes, proper
    authentication, computing resource allocation, etc.. Cleanup for the
    resources does not happen automatically, and you should call `Stop()` to do
    that if you have ever called `Start()`.

    It is not allowed to run `Start()` twice. If you need to restart the job,
    you should create another model server runner instance.
    """

  @abc.abstractmethod
  def WaitUntilRunning(self, deadline: float) -> None:
    """Wait until model server job is running.

    When this method is returned without error, the model server job is in the
    Running state where you can perform all the infra validation logic. It does
    not guarantee that model server job would remain in the Running state
    forever, (e.g. preemption could happen in some serving platform) and any
    kind of infra validation logic failure can be caused from model server job
    not being in the Running state. Still, it is a validation failure and we
    blame model for this.

    Args:
      deadline: A deadline time in UTC timestamp (in seconds).
    Returns:
      Whether the model is available or not.
    """

  @abc.abstractmethod
  def Stop(self) -> None:
    """Stop the model server in blocking manner.

    Model server job would be gracefully stopped once infra validation logic is
    done. Here is the place you need to cleanup every resources you've created
    in the `Start()`. It is recommended not to raise error during the `Stop()`
    as it will usually be called in the `finally` block.

    `Stop()` is guaranteed to be called if `Start()` is ever called, unless the
    process dies unexpectedly due to external factors (e.g. SIGKILL). `Stop()`
    can be called even when `Start()` was not completed. `Stop()` should not
    assume the completion of `Start()`.

    `Stop()` is also called when graceful shutdown for the *executor* (not
    model server) is requested. `Stop()` method should be finished within the
    graceful shutdown period, and it is perfectly fine to add a retry logic
    inside `Stop()` until the deadline is met.
    """
