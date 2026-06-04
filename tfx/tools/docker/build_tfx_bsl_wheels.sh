#!/bin/bash
# Build tfx-bsl wheels from source with patch applied.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/tfx_bsl_build"
TFX_BSL_REPO="https://github.com/tensorflow/tfx-bsl/"
TFX_BSL_TAG="master"
OUTPUT_DIR="${1:-.}"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Cloning tfx-bsl repository..."
git clone --no-depth "$TFX_BSL_REPO" tfx-bsl
cd tfx-bsl

echo "Checking out to $TFX_BSL_TAG..."
git checkout "$TFX_BSL_TAG"

echo "Loosening tensorflow-serving-api requirements for TF 2.21 compatibility..."
sed -i 's/>=2.19,<2.20/>=2.19,<2.22/g' setup.py

echo "Applying tfx_bsl.patch..."
if [[ -f "$SCRIPT_DIR/tfx_bsl.patch" ]]; then
  git apply "$SCRIPT_DIR/tfx_bsl.patch" || echo "Warning: tfx_bsl.patch could not be applied, skipping..."
fi

echo "Building wheels..."
export USE_BAZEL_VERSION=7.7.0
export LDFLAGS="-fuse-ld=bfd"
pip install numpy==1.26.4
export TFX_DEPENDENCY_SELECTOR=NIGHTLY
CFLAGS=$(python-config --cflags) python setup.py bdist_wheel

echo "Copying wheels to output directory..."
mkdir -p "$OUTPUT_DIR"
cp dist/*.whl "$OUTPUT_DIR/"

echo "Wheels built and copied to $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.whl

echo "Build completed successfully!"
