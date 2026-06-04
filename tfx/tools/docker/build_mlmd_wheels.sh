#!/bin/bash
# Build ml-metadata wheels from source.
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/tmp/mlmd_build"
MLMD_REPO="https://github.com/google/ml-metadata/"
MLMD_TAG="master"
OUTPUT_DIR="${1:-.}"

echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Cloning mlmd repository..."
git clone --no-depth "$MLMD_REPO" ml-metadata
cd ml-metadata

echo "Checking out to $MLMD_TAG..."
git checkout "$MLMD_TAG"

echo "Building wheels..."
export USE_BAZEL_VERSION=7.7.0
export LDFLAGS="-fuse-ld=bfd"
echo "DEBUG: Listing /usr/lib/jvm/:"
ls -la /usr/lib/jvm/ || true
echo "DEBUG: javac location:"
which javac || true
readlink -f /usr/bin/javac || true
export JAVA_HOME=$(readlink -f /usr/bin/javac | sed "s:/bin/javac::")
echo "DEBUG: JAVA_HOME is set to: $JAVA_HOME"
pip install numpy==1.26.4
export TFX_DEPENDENCY_SELECTOR=NIGHTLY
CFLAGS=$(python-config --cflags) python setup.py bdist_wheel

echo "Copying wheels to output directory..."
mkdir -p "$OUTPUT_DIR"
cp dist/*.whl "$OUTPUT_DIR/"

echo "Wheels built and copied to $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/*.whl

echo "Build completed successfully!"
