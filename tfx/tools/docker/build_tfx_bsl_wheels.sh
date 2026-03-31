#!/bin/bash
# Build tfx-bsl wheels from source with patch applied.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/tfx_bsl_build"
TFX_BSL_REPO="https://github.com/tensorflow/tfx-bsl/"
TFX_BSL_TAG="v1.17.1"
OUTPUT_DIR="${1:-.}"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Cloning tfx-bsl repository..."
git clone --no-depth "$TFX_BSL_REPO" tfx-bsl
cd tfx-bsl

echo "Fetching tag $TFX_BSL_TAG..."
git fetch origin tag "$TFX_BSL_TAG"

echo "Checking out to $TFX_BSL_TAG..."
git checkout "$TFX_BSL_TAG"

echo "Applying tfx_bsl.patch..."
if [[ -f "$SCRIPT_DIR/tfx_bsl.patch" ]]; then
  git apply "$SCRIPT_DIR/tfx_bsl.patch"
else
  echo "Error: tfx_bsl.patch not found at $SCRIPT_DIR/tfx_bsl.patch" >&2
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
