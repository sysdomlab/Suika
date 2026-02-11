FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /root/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       ninja-build \
       git \
       curl \
       ca-certificates \
       pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY . /root/workspace/

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir pybind11

RUN python -m pip install --no-cache-dir -e . \
    && cd Suika/cluster/cpp_backend \
    && python setup.py

CMD ["bash"]