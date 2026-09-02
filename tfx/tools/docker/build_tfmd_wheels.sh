#!/bin/bash
# Build tensorflow-metadata wheels from source.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/tfmd_build"
TFMD_REPO="https://github.com/tensorflow/metadata/"
TFMD_TAG="r1.21.0"
OUTPUT_DIR="${1:-.}"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Cloning metadata repository..."
git clone --no-depth "$TFMD_REPO" metadata
cd metadata

echo "Checking out to $TFMD_TAG..."
git checkout "$TFMD_TAG"

echo "Building wheels..."
export USE_BAZEL_VERSION=7.7.0
export LDFLAGS="-fuse-ld=bfd"
pip install numpy==1.26.4
export TFX_DEPENDENCY_SELECTOR=UNCONSTRAINED
CFLAGS=$(python-config --cflags) python setup.py bdist_wheel

echo "Copying wheels to output directory..."
mkdir -p "$OUTPUT_DIR"
cp dist/*.whl "$OUTPUT_DIR/"

echo "Wheels built and copied to $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.whl

echo "Build completed successfully!"
