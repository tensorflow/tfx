FROM python:3.7

# Install test package of TFX and kfp. Since this is an experimental feature,
# we are installing a test version of TFX. The below command
# RUN pip install tfx
# would be sufficient once the feature becomes available
RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.python.org/simple/ tfx-eric
