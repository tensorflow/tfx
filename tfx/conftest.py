"""Test configuration."""
import importlib.util
import os
import sys
import threading
import time
import traceback

# Prioritize the local cloned repository workspace root in sys.path to ensure testdata is resolvable.
_workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _workspace_root in sys.path:
  sys.path.remove(_workspace_root)
sys.path.insert(0, _workspace_root)

def debug_excepthook(exc_type, exc_value, exc_traceback):
  try:
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = "".join(tb_lines)
    os.write(2, b"\n=================== TFX DEBUG EXCEPTHOOK ===================\n")
    os.write(2, tb_text.encode('utf-8'))
    os.write(2, b"============================================================\n\n")
  except Exception as e:
    try:
      os.write(2, f"Failed to write exception in debug_excepthook: {e}\n".encode('utf-8'))
    except Exception:
      pass
  sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = debug_excepthook

# Disable deprecated lookup warnings in Airflow and speed up execution
os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = 'sqlite:////tmp/airflow.db'

# Prevent library thread pool and gRPC fork deadlocks under multi-process/multithreaded environments
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Monkey-patch PipelineOptions to force fast, low-overhead in-memory DirectRunner under unit tests.
try:
  from apache_beam.options.pipeline_options import PipelineOptions

  original_init = PipelineOptions.__init__

  def custom_init(self, flags=None, **kwargs):
    import sys
    if flags is None:
      flags_list = list(sys.argv)
    else:
      flags_list = list(flags)

    has_other_runner = False
    for flag in flags_list:
      if isinstance(flag, str) and flag.startswith('--runner=') and 'DirectRunner' not in flag:
        has_other_runner = True
        break

    runner_kwarg = kwargs.get('runner')
    if runner_kwarg and 'DirectRunner' not in str(runner_kwarg):
      has_other_runner = True

    if not has_other_runner:
      if not any(isinstance(flag, str) and flag.startswith('--direct_running_mode=') for flag in flags_list):
        flags_list.append('--direct_running_mode=in_memory')

    original_init(self, flags=flags_list, **kwargs)

  PipelineOptions.__init__ = custom_init
except Exception:
  pass

from absl import flags  # noqa: E402


def pytest_configure(config):
  # This is needed to avoid
  # `absl.flags._exceptions.UnparsedFlagAccessError` in some tests.
  flags.FLAGS.mark_as_parsed()


def _is_installed(module_name):
  try:
    return importlib.util.find_spec(module_name) is not None
  except Exception:
    return False


def pytest_ignore_collect(collection_path, config):
  path_str = str(collection_path)
  # Ignore Kubeflow/Vertex related tests if kfp is not installed
  if any(k in path_str for k in ('kubeflow', 'kfp', 'vertex', 'penguin_pipeline_sklearn_gcp_test')):
    if not _is_installed('kfp'):
      return True
  # Ignore ranking tests if struct2tensor is not installed/functional
  if 'ranking' in path_str:
    try:
      import struct2tensor  # noqa: F401
    except Exception:
      return True
  # Ignore Airflow related tests if airflow is not installed
  if 'airflow' in path_str or 'chicago_taxi_pipeline/taxi_pipeline_simple_test' in path_str:
    if not _is_installed('airflow'):
      return True
  # Ignore interactive context tests if nbformat is not installed
  if 'interactive_context' in path_str:
    if not _is_installed('nbformat'):
      return True
  return False


# Pure-Python sentinel thread to print tracebacks of all active threads and exit immediately if any test hangs/takes too long
class HangSentinel(threading.Thread):
  def __init__(self, timeout=120):
    super().__init__()
    self.timeout = timeout
    self.daemon = True
    self.last_heartbeat = time.time()
    self.active = True
    self.current_test = "None"

  def heartbeat(self, test_name):
    self.last_heartbeat = time.time()
    self.current_test = test_name

  def run(self):
    while self.active:
      time.sleep(5)
      if time.time() - self.last_heartbeat > self.timeout:
        sys.stderr.write("\n================!!! HANG SENTINEL TIMEOUT DETECTED !!!================\n")
        sys.stderr.write(f"Test '{self.current_test}' has been running for {time.time() - self.last_heartbeat:.1f}s (Threshold: {self.timeout}s)!\n")
        sys.stderr.write("=== ACTIVE THREADS STACK TRACES ===\n")
        for thread_id, frame in sys._current_frames().items():
          thread_name = "Unknown"
          for t in threading.enumerate():
            if t.ident == thread_id:
              thread_name = t.name
              break
          sys.stderr.write(f"\nThread: {thread_name} (ID: {thread_id}):\n")
          tb_lines = traceback.format_stack(frame)
          sys.stderr.write("".join(tb_lines))
        sys.stderr.write("============================================================\n\n")
        sys.stderr.flush()
        time.sleep(2)  # Secure pipe flush delivery to GHA host!
        os._exit(124)

_sentinel = None

def pytest_sessionstart(session):
  global _sentinel
  if 'TEST_TMPDIR' in os.environ or 'TEST_UNDECLARED_OUTPUTS_DIR' in os.environ or os.environ.get('GITHUB_ACTIONS'):
    timeout = 120
    # Increase timeout significantly (15 minutes) if running e2e tests
    for arg in sys.argv:
      if 'e2e' in arg:
        timeout = 900
        break
    _sentinel = HangSentinel(timeout=timeout)
    _sentinel.start()

def pytest_sessionfinish(session, exitstatus):
  global _sentinel
  if _sentinel:
    _sentinel.active = False

def pytest_runtest_setup(item):
  global _sentinel
  if _sentinel:
    _sentinel.heartbeat(f"{item.nodeid} [SETUP]")

def pytest_runtest_call(item):
  global _sentinel
  if _sentinel:
    _sentinel.heartbeat(f"{item.nodeid} [CALL]")

def pytest_runtest_teardown(item):
  global _sentinel
  if _sentinel:
    _sentinel.heartbeat(f"{item.nodeid} [TEARDOWN]")
