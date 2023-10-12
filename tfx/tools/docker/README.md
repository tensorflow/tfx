Files for building the [Docker](http://www.docker.com) image for TFX.

To build a docker image, run below command with flags under root directory of
github checkout. ``` ./tfx/tools/docker/build_docker_image.sh --build-arg
TFX_DEPENDENCY_SELECTOR=NIGHTLY --build-arg
BASE_IMAGE=gcr.io/deeplearning-platform-release/tf2-gpu.2-13.py310 --build-arg
BEAM_VERSION=2.50.0 --build-arg ADDITIONAL_PACKAGES=tensorflow==2.13.0

```

`NOTE:` It is recommended to use images on [tensorflow/tfx docker hub](https://hub.docker.com/r/tensorflow/tfx/tags) or [TFX GCR](https://gcr.io/tfx-oss-public/tfx), instead of building docker image directly.
```
