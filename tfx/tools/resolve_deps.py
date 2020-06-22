# Lint as: python3
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

# TODO(b/158994632): Move this module under opensource_only.
"""Setuptools ResolveDeps command.

Resolve full transitive dependency from dependency constraints in setup.py. If
resolution was successful it prints the list of decided version. If resolution
fails it dies with error, trying its best to output the failure reason.

Currently dependency resolution is backed by Poetry's Mixology (PubGrub
algorithm), which is different from pip's resolver (based on ResolveLib).

Args:
  --tf-version: Directives for selecting tensorflow version. Can be one of
      RELEASED_TF, RELEASED_TF_2, or PRERELEASED_TF_2.

Usage:
  python setup.py resolve_deps
  python setup.py resolve_deps --tf-version RELEASED_TF_2
"""

import enum
import sys

import setuptools


class TFVersionDirective(enum.Enum):
  """Enum for TensorFlow version directive."""
  # Install TF 1.x
  RELEASED_TF = 'RELEASED_TF'
  # Install TF 2.x
  RELEASED_TF_2 = 'RELEASED_TF_2'
  # Install pre-released version of TF 2.x
  PRERELEASED_TF_2 = 'PRERELEASED_TF_2'

  @property
  def is_pre(self):
    return self.value.startswith('PRE')

  @property
  def pep508(self):
    if self.value.endswith('TF'):
      return 'tensorflow<2'
    elif self.value.endswith('TF_2'):
      return 'tensorflow>=2,<3'
    raise AssertionError('Invalid value {}'.format(self.value))


class ResolveDepsCommand(setuptools.Command):
  """Setuptools command for dependency resolution.

  See module docstring for more information.
  """

  user_options = [
      ('tf-version=', None, 'TensorFlow version specifier enums.')
  ]

  def initialize_options(self):
    # pylint: disable=g-import-not-at-top
    # These modules are not available on import time, but available when
    # setup.py command is running.
    from poetry import repositories
    from poetry.repositories import pypi_repository

    self._pool = repositories.Pool()
    self._pool.add_repository(pypi_repository.PyPiRepository())
    self.tf_version = None

  def finalize_options(self):
    self._override_deps = []
    if self.tf_version:
      directive = TFVersionDirective(self.tf_version)
      latest_tf_version = self._find_latest(
          directive.pep508, directive.is_pre)
      self._override_deps.append(
          'tensorflow==' + latest_tf_version)

  def _find_latest(self, pep508, allow_pre):
    # pylint: disable=g-import-not-at-top
    # These modules are not available on import time, but available when
    # setup.py command is running.
    from poetry.version import requirements

    req = requirements.Requirement(pep508)
    result = self._pool.find_packages(
        req.name,
        req.constraint,
        allow_prereleases=allow_pre)
    if not result:
      raise ValueError('No packages found for ' + pep508)
    return max(p.version for p in result).text

  def _parse_deps(self, pep508_list):
    # pylint: disable=g-import-not-at-top
    # These modules are not available on import time, but available when
    # setup.py command is running.
    from poetry import packages

    return [packages.dependency_from_pep_508(v) for v in pep508_list]

  def run(self):
    # pylint: disable=g-import-not-at-top
    # These modules are not available on import time, but available when
    # setup.py command is running.
    import clikit.io
    from poetry import packages
    from poetry.puzzle import provider
    from poetry.mixology import version_solver  # pylint: disable=g-bad-import-order

    # root package is the project package (e.g. tfx) that specify other
    # dependencies. Consider it as a mock library that has other dependencies.
    # It doesn't have to be a real package name nor version, so we're using
    # arbitrary value that is helpful on debugging.
    root = packages.ProjectPackage(
        name='library_under_test',
        version='0.1.0')
    deps = self._parse_deps(self.distribution.install_requires)
    override_deps = self._parse_deps(self._override_deps)
    override_names = {d.name for d in override_deps}
    root.requires = [d for d in deps
                     if d.name not in override_names] + override_deps
    root.python_versions = '=={0}.{1}.{2}'.format(*sys.version_info[:3])
    solver = version_solver.VersionSolver(
        root=root,
        provider=provider.Provider(
            package=root,
            pool=self._pool,
            io=clikit.io.ConsoleIO()))
    solver.solve()
    for pkg in solver.solution.decisions:
      if pkg.name != root.name:
        # TODO(b/158753830): Utilize resolution result rather than printing.
        # Also the print result is mixed with distutils.log (which corrupts
        # stdout as well) and cannot be redirected.
        print('{0}=={1}'.format(pkg.name, pkg.version))
