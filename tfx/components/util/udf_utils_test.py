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
"""Tests for tfx.components.util.udf_utils."""

import hashlib
import os
import subprocess
import sys
import tempfile

from unittest import mock
import tensorflow as tf

from tfx.components.util import udf_utils
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.types import component_spec
from tfx.utils import import_utils


class _MyComponentSpec(component_spec.ComponentSpec):
  PARAMETERS = {
      'my_module_file':
          component_spec.ExecutionParameter(type=str, optional=True),
      'my_module_path':
          component_spec.ExecutionParameter(type=str, optional=True),
  }
  INPUTS = {}
  OUTPUTS = {}


class _MyComponent(base_component.BaseComponent):
  SPEC_CLASS = _MyComponentSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(base_executor.BaseExecutor)


class UdfUtilsTest(tf.test.TestCase):

  @mock.patch.object(import_utils, 'import_func_from_source')
  def testGetFnFromSource(self, mock_import_func):
    exec_properties = {'module_file': 'path/to/module_file.py'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path/to/module_file.py',
                                             'test_fn')

  @mock.patch.object(import_utils, 'import_func_from_module')
  def testGetFnFromModule(self, mock_import_func):
    exec_properties = {'module_path': 'path.to.module'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path.to.module', 'test_fn')

  @mock.patch.object(import_utils, 'import_func_from_module')
  def testGetFnFromModuleFn(self, mock_import_func):
    exec_properties = {'test_fn': 'path.to.module.test_fn'}
    udf_utils.get_fn(exec_properties, 'test_fn')
    mock_import_func.assert_called_once_with('path.to.module', 'test_fn')

  def testGetFnFailure(self):
    with self.assertRaises(ValueError):
      udf_utils.get_fn({}, 'test_fn')

  def test_ephemeral_setup_py_contents(self):
    contents = udf_utils._get_ephemeral_setup_py_contents(
        'my_pkg', '0.0+xyz', ['a', 'abc', 'xyz'])
    self.assertIn("name='my_pkg',", contents)
    self.assertIn("version='0.0+xyz',", contents)
    self.assertIn("py_modules=['a', 'abc', 'xyz'],", contents)

  def test_version_hash(self):

    def _write_temp_file(user_module_dir, file_name, contents):
      with open(os.path.join(user_module_dir, file_name), 'w') as f:
        f.write(contents)

    user_module_dir = tempfile.mkdtemp()
    _write_temp_file(user_module_dir, 'a.py', 'aa1')
    _write_temp_file(user_module_dir, 'bb.py', 'bbb2')
    _write_temp_file(user_module_dir, 'ccc.py', 'cccc3')
    _write_temp_file(user_module_dir, 'dddd.py', 'ddddd4')

    expected_plaintext = (
        # Length and encoding of "a.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x04a.py'
        # Length and encoding of contents of "a.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x03aa1'
        # Length and encoding of "ccc.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x06ccc.py'
        # Length and encoding of contents of "ccc.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x05cccc3'
        # Length and encoding of "dddd.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x07dddd.py'
        # Length and encoding of contents of "dddd.py".
        b'\x00\x00\x00\x00\x00\x00\x00\x06ddddd4')
    h = hashlib.sha256()
    h.update(expected_plaintext)
    expected_version_hash = h.hexdigest()
    self.assertEqual(
        expected_version_hash,
        '4fecd9af212c76ee4097037caf78c6ba02a2e82584837f2031bcffa0f21df43e')
    self.assertEqual(
        udf_utils._get_version_hash(user_module_dir,
                                    ['dddd.py', 'a.py', 'ccc.py']),
        expected_version_hash)

  def testAddModuleDependencyAndPackage(self):
    # Do not test packaging in unsupported environments.
    if not udf_utils.should_package_user_modules():
      return

    # Create a component with a testing user module file.
    temp_dir = tempfile.mkdtemp()
    temp_module_file = os.path.join(temp_dir, 'my_user_module.py')
    with open(temp_module_file, 'w') as f:
      f.write('# Test user module file.\nEXPOSED_VALUE="ABC123xyz"')
    component = _MyComponent(
        spec=_MyComponentSpec(my_module_file=temp_module_file))

    # Add the user module file pip dependency.
    udf_utils.add_user_module_dependency(component, 'my_module_file',
                                         'my_module_path')
    self.assertLen(component._pip_dependencies, 1)
    dependency = component._pip_dependencies[0]
    self.assertIsInstance(dependency, udf_utils.UserModuleFilePipDependency)
    self.assertIs(dependency.component, component)
    self.assertEqual(dependency.module_file_key, 'my_module_file')
    self.assertEqual(dependency.module_path_key, 'my_module_path')

    # Resolve the pip dependency and package the user module.
    temp_pipeline_root = tempfile.mkdtemp()
    component._resolve_pip_dependencies(temp_pipeline_root)
    self.assertLen(component._pip_dependencies, 1)
    dependency = component._pip_dependencies[0]

    # The hash version is based on the module names and contents and thus
    # should be stable.
    self.assertEqual(
        dependency,
        os.path.join(
            temp_pipeline_root, '_wheels', 'tfx_user_code_MyComponent-0.0+'
            '1c9b861db85cc54c56a56cbf64f77c1b9d1ded487d60a97d082ead6b250ee62c'
            '-py3-none-any.whl'))

    # Test import behavior within context manager.
    with udf_utils.TempPipInstallContext([dependency]):
      # Test import from same process.
      import my_user_module  # pylint: disable=g-import-not-at-top
      self.assertEqual(my_user_module.EXPOSED_VALUE, 'ABC123xyz')
      del sys.modules['my_user_module']

      # Test import from a subprocess.
      self.assertEqual(
          subprocess.check_output([
              sys.executable, '-c',
              'import my_user_module; print(my_user_module.EXPOSED_VALUE)'
          ]), b'ABC123xyz\n')

    # Test that the import paths are cleaned up, so the user module can no
    # longer be imported.
    with self.assertRaises(ModuleNotFoundError):
      import my_user_module  # pylint: disable=g-import-not-at-top


if __name__ == '__main__':
  tf.test.main()
