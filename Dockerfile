FROM debian:11-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /home/app
COPY yolo/. /home/app/yolo/
RUN rm /home/app/yolo/libdarknet.so
RUN rm /home/app/yolo/experiments -rf
RUN rm /home/app/yolo/result -rf

COPY serverless/* /home/app/



RUN apt-get update \
  && apt-get install --no-install-recommends --no-install-suggests -y git build-essential tesseract-ocr ca-certificates python3 python3-pip
RUN git clone https://github.com/AlexeyAB/darknet.git && cd darknet \
  && sed -i 's/OPENCV=0/OPENCV=0/' Makefile \
  && sed -i 's/GPU=0/GPU=0/' Makefile \
  && sed -i 's/CUDNN=0/CUDNN=0/' Makefile \
  && sed -i 's/CUDNN_HALF=0/CUDNN_HALF=0/' Makefile \
  && sed -i 's/LIBSO=0/LIBSO=1/' Makefile \
  && make && mv libdarknet.so /home/app/yolo/libdarknet.so && cd .. && rm darknet -fr

RUN pip install --upgrade pip
RUN pip install -r /home/app/requirements.txt

WORKDIR /home/app/
ADD https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie /usr/bin/aws-lambda-rie
RUN apt autoremove -y git build-essential ca-certificates
RUN chmod 755 /usr/bin/aws-lambda-rie /home/app/entry.sh
ENTRYPOINT [ "/home/app/entry.sh" ]
CMD [ "app.handler" ]
