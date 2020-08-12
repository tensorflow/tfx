#!/bin/bash -e
# Copyright 2018 Google LLC
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


# A script to gather licenses of locally-installed python packages.
# There are two sources of licenses
# - Locally available LICENSE files which are installed with the package.
# - A license table (3 columns: name,license_link,license_type) csv file
#   which can be used to download license files.
#
# These license files will be copied into specified directory.
# Usage:
#   license.sh third_party_licenses.csv /usr/licenses

set -e
set +x

# Get the list of python packages installed locally.
OLD_IFS="$IFS"
IFS=$'\n'
INSTALLED_PACKAGES_WITH_VERSION=($(pip freeze --exclude-editable | grep -v "pkg-resources"))

# Get the list of python packages tracked in the given CSV file.
REGISTERED_PACKAGES=()
while IFS=, read -r col1 col2 col3
do
  REGISTERED_PACKAGES+=($col1)
done < $1
IFS="$OLD_IFS"

mkdir -p $2

GPL_PACKAGES=()
REMOTE_LICENSE_PACKAGES=()
# Copy locally available LICENSE files and gather packages without them.
for i in "${INSTALLED_PACKAGES_WITH_VERSION[@]}"; do
  # `i` looks like "pkg_name==0.0.1".
  pkg=${i%==*}
  pkg_base_dir=$(python -m pip show ${pkg} | sed -n 's/^Location: //p')
  # LICENSE files are located under .../site-packages/pkg_name-0.0.1.dist-info/.
  pkg_without_underscore=${i//-/_}
  pkg_dist_info_dir="${pkg_base_dir}/${pkg_without_underscore/==/-}.dist-info"
  set +e
  license_files=$(ls ${pkg_dist_info_dir}/LICENSE* 2> /dev/null)
  if [[ $? -ne 0 ]]; then
    REMOTE_LICENSE_PACKAGES+=("$pkg")
  else
    echo "License found: ${license_files}"
    cat ${license_files} > $2/$pkg.LICENSE
  fi
  set -e

  license=$(python -m pip show ${pkg} | sed -n 's/^License: //p')
  if [[ "${license}" == *GPL* ]]; then
    GPL_PACKAGES+=("$pkg")
  fi
done

# Collect missing packages.
MISSING=()
for i in "${REMOTE_LICENSE_PACKAGES[@]}"; do
  skip=
  for j in "${REGISTERED_PACKAGES[@]}"; do
    # PyPI package name is case-insensitive so cast to lower case.
    [[ ${i,,} == ${j,,} ]] && { skip=1; break; }
  done
  [[ -n $skip ]] || MISSING+=("$i")
done

if [ -n "$MISSING" ]; then
  echo "The following packages are not found for licenses tracking."
  echo "Please add an entry in $1 for each of them."
  echo ${MISSING[@]}
  exit 1
fi

# Download licenses from the url in the csv file.
# This should be done before guessing with github URL to override the problem of
# packages with wrong(but working) Home-page URLs.
while IFS=, read -r col1 col2 col3
do
  if [[ " ${REMOTE_LICENSE_PACKAGES[@]} " =~ " ${col1} " ]]; then
    echo "Downloading license for ${col1} from ${col2}"
    curl --fail -sSL -o $2/$col1.LICENSE $col2
  fi
done < $1
IFS="$OLD_IFS"

# Download source code for GPL packages.
mkdir -p $2/source
for i in "${GPL_PACKAGES[@]}"; do
  echo "Downloading source of the GPL-licensed package: ${i}"
  python -m pip install -t "$2/source/${i}" ${i}
done
