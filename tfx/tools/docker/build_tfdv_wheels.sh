#!/bin/bash
# Build tensorflow-data-validation wheels from source with patch applied.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/tfdv_build"
TFDV_REPO="https://github.com/tensorflow/data-validation/"
TFDV_TAG="v1.17.0"
OUTPUT_DIR="${1:-.}"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Cloning data-validation repository..."
git clone --no-depth "$TFDV_REPO" data-validation
cd data-validation

echo "Fetching tag $TFDV_TAG..."
git fetch origin tag "$TFDV_TAG"

echo "Checking out to $TFDV_TAG..."
git checkout "$TFDV_TAG"

echo "Applying tfdv.patch..."
if [[ -f "$SCRIPT_DIR/tfdv.patch" ]]; then
  git apply "$SCRIPT_DIR/tfdv.patch"
else
  echo "Error: tfdv.patch not found at $SCRIPT_DIR/tfdv.patch" >&2
  exit 1
fi

echo "Building wheels..."
export USE_BAZEL_VERSION=6.5.0
export LDFLAGS="-fuse-ld=bfd"
pip install numpy==1.24.4
CFLAGS=$(python-config --cflags) python setup.py bdist_wheel

echo "Copying wheels to output directory..."
mkdir -p "$OUTPUT_DIR"
cp dist/*.whl "$OUTPUT_DIR/"

echo "Wheels built and copied to $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.whl

echo "Build completed successfully!"
