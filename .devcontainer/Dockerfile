FROM arm64v8/ubuntu:focal

RUN apt-get update
RUN apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR="/usr/local/bin/miniconda3"
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN wget \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
      -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $CONDA_DIR \
    && rm -f ~/miniconda.sh

COPY environment.yaml requirements.txt /tmp/conda-tmp/
RUN umask 0002 \ 
    && conda env update -n base -f /tmp/conda-tmp/environment.yaml \
    && rm -rf /tmp/conda-tmp

RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1001 user
USER user
WORKDIR /home/user

RUN mkdir /home/user/transductions
WORKDIR /home/user/transductions 
