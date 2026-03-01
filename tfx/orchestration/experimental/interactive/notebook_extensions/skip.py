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
"""Custom magic for marking cells to be skipped during pipeline export."""

from IPython.core.magic import cell_magic
from IPython.core.magic import Magics
from IPython.core.magic import magics_class


@magics_class
class SkipMagics(Magics):

  @cell_magic
  def skip_for_export(self, line, cell):
    # Execute the cell normally for now. During export to pipeline, this cell
    # will be skipped.
    self.shell.run_cell(cell)
    print('This cell will be skipped during export to pipeline.')


def load_ipython_extension(ipython):
  ipython.register_magics(SkipMagics)
