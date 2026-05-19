"""Test configuration."""
from absl import flags

def pytest_configure(config):
  # This is needed to avoid
  # `absl.flags._exceptions.UnparsedFlagAccessError` in some tests.
  flags.FLAGS.mark_as_parsed()


def pytest_ignore_collect(collection_path, config):
  path_str = str(collection_path)
  # Ignore Kubeflow related tests if kfp is not installed
  if 'kubeflow' in path_str or 'kfp' in path_str:
    try:
      import kfp  # noqa: F401
    except ImportError:
      return True
  # Ignore ranking tests if struct2tensor is not installed/functional
  if 'ranking' in path_str:
    try:
      import struct2tensor  # noqa: F401
    except Exception:
      return True
  return False
