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
"""setup.py file required for Cloud Dataflow Beam runners."""
import setuptools

REQUIRED_PACKAGES = ["tfx==0.21.0",
                     "tensorflow-transform==0.21.2",
                     "tensorflow-model-analysis==0.21.4",
                     "google-cloud-storage==1.29.0",
                     "apache-beam==2.22.0"]

# REQUIRED_PACKAGES = ["tfx==0.21.0",
#                      "tensorflow-transform==0.21.2",
#                      "tensorflow-model-analysis==0.21.4",
#                      "apache-beam==2.18.0"]

setuptools.setup(
    name='cloud_dataflow_pipeline',
    version='0.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages()
)
