# Initialization script for building TFX SDK release packages.
#
# After this script is run, `python setup.py` commands can be run in the
# `tfx/` and `ml-pipelines-sdk/` packages.

BASEDIR=$(dirname "$(pwd)/${0#./}")/..

for CONFIG_NAME in tfx ml-pipelines-sdk
do
  ln -sf $BASEDIR/setup.py $BASEDIR/package_build/$CONFIG_NAME/
  ln -sf $BASEDIR/dist $BASEDIR/package_build/$CONFIG_NAME/
  ln -sf $BASEDIR/tfx $BASEDIR/package_build/$CONFIG_NAME/
  ln -sf $BASEDIR/README*.md $BASEDIR/package_build/$CONFIG_NAME/

  rm -rf $BASEDIR/package_build/$CONFIG_NAME/build
  mkdir $BASEDIR/package_build/$CONFIG_NAME/build
  ln -sf $BASEDIR/build/BUILD $BASEDIR/package_build/$CONFIG_NAME/build/
  ln -sf $BASEDIR/build/gen_proto.sh $BASEDIR/package_build/$CONFIG_NAME/build/
done