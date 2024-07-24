FROM python:3.9-slim

WORKDIR /usr/src/app/

RUN apt update
RUN apt install nano -y
RUN apt install vim -y
RUN apt install screen -y
RUN apt install curl -y
RUN apt install git -y
RUN apt install gcc g++ build-essential -y
RUN apt install libopencv-dev -y
RUN apt install cmake protobuf-compiler -y
RUN apt install libopenblas-dev  -y
RUN apt install python3-pyproj -y

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

RUN git clone https://github.com/sybrenstuvel/Python-RVO2

WORKDIR /usr/src/app/Python-RVO2
RUN python setup.py build
RUN python setup.py install

WORKDIR /usr/src/app/

COPY *.py ./
COPY *.sh ./
COPY core ./core

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]

