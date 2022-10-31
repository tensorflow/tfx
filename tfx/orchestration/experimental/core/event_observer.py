# Copyright 2022 Google LLC. All Rights Reserved.
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
"""event_observer is a module for registering observers to observe events.

This is designed to be used in a with block, e.g.

  with event_observer.init():
    event_observer.register_observer(...)
    event_observer.notify(...)

All calls occurring within the with block (or while the context is active) will
use the same singleton _EventObserver. register_observer(), notify() are
thread-compatible, and support being called from multiple threads. They will
silently have no effect if used outside an active init() context.
"""

from concurrent import futures
import contextlib
import dataclasses
import queue
import threading
from typing import Any, Callable, List, Optional, Union

from absl import logging
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


@dataclasses.dataclass(frozen=True)
class PipelineStarted:
  """PipelineStarted event."""
  pipeline_id: str
  # Should be pipeline_state.PipelineState, but importing pipeline_state
  # would introduce a circular dependency
  pipeline_state: Any


@dataclasses.dataclass(frozen=True)
class PipelineFinished:
  """PipelineFinished event."""
  pipeline_id: str
  # Should be pipeline_state.PipelineState, but importing pipeline_state
  # would introduce a circular dependency
  pipeline_state: Any
  status: status_lib.Status


@dataclasses.dataclass(frozen=True)
class NodeStateChange:
  """NodeStateChange event."""
  execution: metadata_store_pb2.Execution
  pipeline_id: str
  pipeline_run: str
  node_id: str
  # old_state and new_state are of type NodeState, but we can't refer to that
  # type without either introducing a circular dependency (if we refer to
  # NodeState via pipeline_state), or breaking backwards compatibility (if we
  # move the NodeState type to its own module) due to the fully qualified type
  # name being serialised as part of the JSON encoding for all
  # json_utils.Jsonable types.
  old_state: Any
  new_state: Any


Event = Union[PipelineStarted, PipelineFinished, NodeStateChange]

ObserverFn = Callable[[Event], None]


def register_observer(observer_fn: ObserverFn) -> None:
  """Register an observer.

  Registers an observer. The observer function will be called whenever an event
  triggers.

  Silently does nothing if not in an init() context.

  Args:
    observer_fn: A function that takes in an Event.
  """
  global _event_observer
  global _event_observer_lock
  with _event_observer_lock:
    if _event_observer:
      _event_observer.register_observer(observer_fn)


def notify(event: Event) -> None:
  """Notify that an event occurred.

  Silently does nothing if not in an init() context.

  Args:
    event: Event that occurred.
  """
  global _event_observer
  global _event_observer_lock
  with _event_observer_lock:
    if _event_observer:
      _event_observer.notify(event)


def check_active() -> None:
  """Checks that the main _EventObserver observer thread is active.

  Silently does nothing if not in an init() context.
  """
  global _event_observer
  global _event_observer_lock
  with _event_observer_lock:
    if _event_observer:
      if _event_observer.done():
        ex = _event_observer.exception()
        if ex:
          raise ValueError("_EventObserver observer thread unexpectedly "
                           "terminated with exception") from ex
        else:
          raise ValueError("_EventObserver observer thread unexpectedly "
                           "terminated, but with no exception")


def testonly_wait() -> None:
  global _event_observer
  global _event_observer_lock
  with _event_observer_lock:
    if not _event_observer:
      raise RuntimeError(
          "testonly_wait should only be called in an active init() context")
    _event_observer.testonly_wait()


_event_observer = None
_event_observer_lock = threading.Lock()


@contextlib.contextmanager
def init():
  """Initialises the singleton _EventObserver.

  register_observer() and notify() will use the singleton _EventObserver while
  within this context. The singleton EventObserver will be initialised on
  entering this context, and shut down on exiting this context.

  Raises:
    RuntimeError: If this context is invoked again when it is already active.

  Yields:
    Nothing.
  """
  global _event_observer
  global _event_observer_lock

  with _event_observer_lock:
    if _event_observer is not None:
      raise RuntimeError("nested calls to init() are prohibited")
    _event_observer = _EventObserver()
    _event_observer.start()

  try:
    yield
  finally:
    with _event_observer_lock:
      _event_observer.shutdown()
      _event_observer = None


class _EventObserver:
  """EventObserver.

  Users should only call the module-level functions.  Methods in this class
  should only be invoked by functions in this module.

  Events are guaranteed to be observed in the order they were notify()-ed.

  Observer functions *may* be called in any order (even though the current
  implementation calls them in the registration order, this may change).

  Observer functions *may* be called concurrently (even though the current
  implementation calls them serially, this may change).

  Exceptions in the observer functions are logged, but ignored. Note that a
  slow or stuck observer function may cause events to stop getting observed
  (which is why we may switch to calling them concurrently / with a timeout
  in the future).
  """
  _event_queue: queue.Queue
  _observers: List[ObserverFn]
  _observers_lock: threading.Lock
  _executor: futures.ThreadPoolExecutor

  def __init__(self):
    """_EventObserver constructor."""
    self._event_queue = queue.Queue()
    self._observers = []
    self._observers_lock = threading.Lock()
    self._shutdown_event = threading.Event()
    self._main_executor = futures.ThreadPoolExecutor(max_workers=1)
    self._main_future = None

  def start(self):
    # Not thread-safe. Should only be called from a single thread.
    if self._main_future is not None:
      raise RuntimeError("_EventObserver already started")
    if self._shutdown_event.is_set():
      raise RuntimeError("_EventObserver already shut down")
    self._main_future = self._main_executor.submit(self._main)

  def done(self) -> bool:
    """Returns `True` if the main observation thread has exited.

    Raises:
      RuntimeError: If `done` is called while this _EventObserver isn't in an
        active state.
    """
    if self._main_future is None:
      raise RuntimeError("_EventObserver not in an active state")
    return self._main_future.done()

  def exception(self) -> Optional[BaseException]:
    """Returns exception raised by the main observation thread (if any).

    Raises:
      RuntimeError: If `exception` called while this _EventObserver isn't in an
        active state, or if the main thread is not done (`done` returns
        `False`).
    """
    if self._main_future is None:
      raise RuntimeError("_EventObserver not in an active state")
    if not self._main_future.done():
      raise RuntimeError("Main observation thread not done; call should be "
                         "conditioned on `done` returning `True`.")
    return self._main_future.exception()

  def shutdown(self):
    # Not thread-safe. Should only be called from a single thread.
    if self._shutdown_event.is_set():
      raise RuntimeError("_EventObserver already shut down")
    if self._main_future is None:
      raise RuntimeError("_EventObserver not started")
    self._shutdown_event.set()
    self._main_executor.shutdown()
    self._main_future = None

  def register_observer(self, observer_fn: ObserverFn) -> None:
    with self._observers_lock:
      self._observers.append(observer_fn)

  def notify(self, event: Event) -> None:
    with self._observers_lock:
      if not self._observers:
        return
    self._event_queue.put(event)

  def testonly_wait(self) -> None:
    """Wait for all existing events in the queue to be observed.

    For use in tests only.
    """
    self._event_queue.join()

  def _main(self) -> None:
    """Main observation loop. Checks event queue for events, calls observers."""

    def observe_event(event):
      with self._observers_lock:
        observers = self._observers[:]
      for observer_fn in observers:
        try:
          observer_fn(event)
        except Exception as e:  # pylint: disable=broad-except
          logging.exception(
              "Exception raised by observer function when observing "
              "event %s: %s", event, e)

    def dequeue():
      try:
        return self._event_queue.get(block=True, timeout=5)
      except queue.Empty:
        return None

    while not self._shutdown_event.is_set():
      event = dequeue()
      if event is not None:
        observe_event(event)
        self._event_queue.task_done()
