FROM tensorflow/tfx

COPY requirements.txt . 

ENV AIRFLOW_HOME "/home/airflow"

RUN pip install --no-cache -r requirements.txt

# removing base script
RUN rm -rf * 

WORKDIR /home/

# RUN airflow initdb

COPY setup.sh /

ENTRYPOINT bash -c "/setup.sh"
