FROM phusion/baseimage:focal-1.0.0alpha1-amd64 as base
MAINTAINER Farkhad Kuanyshkereyev, farkhad.kuanyshkereyev@gmail.com

COPY . /usr/src/app
WORKDIR /usr/src/app

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Almaty \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    VIRTUAL_ENV=/opt/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    python3.8-dev \
    software-properties-common \
    gdal-bin \
    libgdal-dev \
    swig \
    potrace \
    wget \
    unzip \
    file \
    curl \
    libpq-dev \
    libspatialindex-dev \
    libsm6 libxext6 libxrender-dev ffmpeg libgl1-mesa-dev \
    python3-pip python3-venv \
    libeccodes0 &&\
#    nodejs \
#    npm &&\
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
    python3 -m venv $VIRTUAL_ENV

RUN pip3 install -U pip wheel --no-cache-dir setuptools==58.0 numpy &&\
    pip3 install --global-option=build_ext \
                --global-option="-I/usr/include/gdal" \
                GDAL==$(gdal-config --version) &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#RUN npm i -g simplify-geojson

FROM base as builder
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ansible sshpass \
     &&\
    pip3 install poetry ansible &&\
    ansible-galaxy install lgblkb.lgblkb_deployer

FROM base as production
RUN mkdir -p /usr/src/app/caches /usr/src/app/data
RUN pip3 install --no-cache-dir -r requirements.txt