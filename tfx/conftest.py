"""Test configuration."""
import importlib.util
import os
import sys
import traceback

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
