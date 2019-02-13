#!/usr/bin/env bash
# Copyright 2018 Google LLC
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

# This scripts starts a worker for for beam. Users should not call this script directly.
# This script depends on Apache Beam code base. Please checkout the Apache Beam codebase and run the following gradle command from Apache Beam checkout location.
# `./gradlew :beam-sdks-python:createProcessWorker`
# Make sure to replace the following variables with appropriate values.

BEAM_CLONE_LOCATION="<BEAM CHECKOUT LOCATION>"
VIRTUAL_ENV="<VIRTUAL ENVIRONMENT WILL ALL THE REQUIREMENTS INSTALLED>"

sh -c "pip=`which pip` . $VIRTUAL_ENV/bin/activate && $BEAM_CLONE_LOCATION/sdks/python/container/build/target/launcher/linux_amd64/boot $* "
