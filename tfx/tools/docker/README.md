Files for building the [Docker](https://hub.docker.com/r/tensorflow/tfx/tags) image for TFX.

To build a docker image, run below command with flags under root directory of
github checkout.
```
./tfx/tools/docker/build_docker_image.sh  --build-arg TFX_DEPENDENCY_SELECTOR=DEFAULT
```

```

`NOTE:` It is recommended to use images on [tensorflow/tfx docker hub](https://hub.docker.com/r/tensorflow/tfx/tags) or [TFX GCR](https://gcr.io/tfx-oss-public/tfx), instead of building docker image directly.
```
