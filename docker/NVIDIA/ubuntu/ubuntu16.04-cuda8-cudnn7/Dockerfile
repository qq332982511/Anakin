FROM nvidia/cuda:10.0-devel-ubuntu18.04

# anakin install ubuntu GPU env
RUN apt-get update ; apt-get install -y\
        build-essential \
        cmake \
        git \
        libiomp-dev \
        libopencv-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-doc \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
        wget 


# install protobuf
RUN wget --no-check-certificate https://mirror.sobukus.de/files/src/protobuf/protobuf-cpp-3.4.0.tar.gz \
                        && tar -xvf protobuf-cpp-3.4.0.tar.gz \
                        && cd protobuf-3.4.0 && ./configure CFLAGS="-fPIC"  CXXFLAGS="-fPIC" \
                        && make -j4 && make install && cd .. && rm -f protobuf-cpp-3.4.0.tar.gz

# set env
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

ENV CUDNN_ROOT=/usr/local/cuda/include

# build and install anakin
