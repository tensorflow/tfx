FROM bumbleblo/official_build_tfx:latest

COPY requirements.txt . 

ENV AIRFLOW_HOME "/home/airflow"

RUN pip3 install --no-cache -r requirements.txt

# removing base script
RUN rm -rf * 

WORKDIR /home/

# RUN airflow initdb

COPY setup.sh /

RUN chmod +x /setup.sh

ENTRYPOINT bash -c "/setup.sh"
