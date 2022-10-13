FROM public.ecr.aws/ubuntu/ubuntu:20.04_stable

LABEL maintainer="Amazon AI"
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

ARG PYTHON=python3
ARG PIP=pip3
ARG TFS_SHORT_VERSION=2.1
ARG TFS_URL=https://tensorflow-aws.s3-us-west-2.amazonaws.com/2.1/Serving/CPU-WITH-MKL/tensorflow_model_server

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8
# Python won’t try to write .pyc or .pyo files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SAGEMAKER_TFS_VERSION="${TFS_SHORT_VERSION}"
ENV PATH="$PATH:/sagemaker"
ENV LD_LIBRARY_PATH='/usr/local/lib:$LD_LIBRARY_PATH'
ENV MODEL_BASE_PATH=/models
# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model
ENV DEBIAN_FRONTEND=noninteractive

# nginx + njs
RUN apt-get update \
 && apt-get -y install --no-install-recommends \
    curl \
    gnupg2 \
    ca-certificates \
    git \
    wget \
    vim \
    build-essential \
    zlib1g-dev \
 && curl -s http://nginx.org/keys/nginx_signing.key | apt-key add - \
 && echo 'deb http://nginx.org/packages/ubuntu/ bionic nginx' >> /etc/apt/sources.list \
 && apt-get update \
 && apt-get -y install --no-install-recommends \
    nginx  

RUN apt-get -y install --no-install-recommends \    
    #nginx-module-njs \
    python3 \
    python3-pip \
    python3-setuptools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN ${PIP} --no-cache-dir install --upgrade pip setuptools


# cython, falcon, gunicorn, grpc
RUN ${PIP} install --no-cache-dir \
    awscli \
    boto3 \
    cython==0.29.14 \
    falcon==2.0.0 \
    gunicorn==20.0.4 \
    #gevent==1.4.0 \
    requests==2.22.0 \
    grpcio==1.27.1 \
    protobuf==3.11.1 \
# using --no-dependencies to avoid installing tensorflow binary
 && ${PIP} install --no-dependencies --no-cache-dir \
    tensorflow-serving-api==2.1.0 
 
#RUN pip install keras==2.9.0


# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN curl https://tensorflow-aws.s3-us-west-2.amazonaws.com/MKL-Libraries/libiomp5.so -o /usr/local/lib/libiomp5.so
RUN curl https://tensorflow-aws.s3-us-west-2.amazonaws.com/MKL-Libraries/libmklml_intel.so -o /usr/local/lib/libmklml_intel.so

RUN curl $TFS_URL -o /usr/bin/tensorflow_model_server \
 && chmod 555 /usr/bin/tensorflow_model_server

# Expose ports
# gRPC and REST
EXPOSE 8500 8501

# Set where models should be stored in the container
RUN mkdir -p ${MODEL_BASE_PATH}

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line
RUN echo '#!/bin/bash \n\n' > /usr/bin/tf_serving_entrypoint.sh \
 && echo '/usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} "$@"' >> /usr/bin/tf_serving_entrypoint.sh \
 && chmod +x /usr/bin/tf_serving_entrypoint.sh

ADD https://raw.githubusercontent.com/aws/aws-deep-learning-containers-utils/master/deep_learning_container.py /usr/local/bin/deep_learning_container.py

RUN chmod +x /usr/local/bin/deep_learning_container.py

RUN curl https://aws-dlc-licenses.s3.amazonaws.com/tensorflow-2.1/license.txt -o /license.txt


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install pandas 
RUN pip install tensorflow
RUN pip install joblib
RUN pip install flask

ENV PATH="/opt/program:${PATH}"
COPY program /opt/program 
WORKDIR /opt/program


