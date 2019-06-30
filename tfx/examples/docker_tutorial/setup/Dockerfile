FROM tensorflow/tfx

RUN pip install --no-cache \
    tensorflow==1.13.1 \
    tfx==0.13.0 \
    apache-airflow \
    jupyter

RUN rm -rf * 

WORKDIR /home

RUN airflow initdb

COPY setup.sh . 

ENTRYPOINT bash -c "./setup.sh"
