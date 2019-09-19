# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup dependencies for local and cloud deployment."""
import setuptools

# LINT.IfChange
TF_VERSION = '1.14.0'
# LINT.ThenChange(train_aiplatform.sh, start_model_server_aiplatform.sh)

# LINT.IfChange
BEAM_VERSION = '2.14.0'
# LINT.ThenChange(setup_beam.sh)

if __name__ == '__main__':
  setuptools.setup(
      name='tfx_chicago_taxi',
      version='0.14.0',
      packages=setuptools.find_packages(),
      install_requires=[
          # TODO(b/139747527) loosen restriction when Beam artifacts can be
          # fetched via Python SDK
          'apache-beam[gcp]==' + BEAM_VERSION,
          'jupyter>=1.0,<2',
          'notebook>=5.7.8,<5.8',
          'numpy>=1.14.5,<2',
          'protobuf>=3.7.0,<3.8.0',
          'tensorflow>=' + TF_VERSION + ',<2',
          'tensorflow-data-validation>=0.14,<0.15',
          'tensorflow-metadata>=0.14,<0.15',
          'tensorflow-model-analysis>=0.14,<0.15',
          'tensorflow-serving-api>=1.14.0,<1.15',
          'tensorflow-transform>=0.14.0,<0.15',
      ],
      python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
  )
