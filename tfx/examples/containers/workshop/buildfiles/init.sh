#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set up the environment for the TFX tutorial

# Create the user
adduser tfx_user --disabled-password
echo "tfx_user:tfx" | chpasswd
adduser tfx_user sudo

# Add sources for GCC and Python3.6
add-apt-repository -y ppa:ubuntu-toolchain-r/test
add-apt-repository -y ppa:jonathonf/python-3.6
apt-get update

# Fix GCC problem with Airflow
apt-get install -y --no-install-recommends gcc-7 g++-7 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

# Upgrade to Python 3.6
apt-get install -y --no-install-recommends python3.6 python3.6-dev

# Download Flink
cd /home/tfx_user
wget http://us.mirrors.quenda.co/apache/flink/flink-1.8.1/flink-1.8.1-bin-scala_2.11.tgz
chown tfx_user:tfx_user flink-1.8.1-bin-scala_2.11.tgz

echo "cd" >> /home/tfx_user/.profile
