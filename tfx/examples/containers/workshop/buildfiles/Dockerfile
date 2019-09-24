FROM ubuntu:16.04
LABEL maintainer="tensorflow-extended-dev@googlegroups.com"

COPY init.sh /
COPY setup_demo.sh /

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        python-pip \
        python3-pip \
        python3-dev \
        virtualenv \
        git \
        sudo \
        vim \
        wget \
        curl \
        build-essential \
        software-properties-common \
        default-jre \
        && apt-get autoclean \
        && apt-get autoremove --purge \
        && chmod +x /init.sh \
        && bash /init.sh

EXPOSE 8080 8888 6006
