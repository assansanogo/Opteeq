FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install --no-install-recommends --no-install-suggests -y git build-essential libopencv-dev
RUN git clone https://github.com/AlexeyAB/darknet.git && cd darknet \
  && sed -i 's/OPENCV=0/OPENCV=1/' Makefile \
  && sed -i 's/GPU=0/GPU=1/' Makefile \
  && sed -i 's/CUDNN=0/CUDNN=1/' Makefile \
  && sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile \
  && sed -i 's/LIBSO=0/LIBSO=1/' Makefile \
  && sed -i 's/# ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]/ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]/' Makefile \
  && make \
  && cp darknet /usr/local/bin \
  && cd .. && rm -rf darknet
