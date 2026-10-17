FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Use a domestic apt mirror; install only wget
RUN apt-get update && apt-get install -y wget git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (Tsinghua mirror)
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_TOS_ACCEPTED=true

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Configure the Tsinghua pip mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Create the environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Activate the environment
ENV CONDA_DEFAULT_ENV=GTNM
ENV PATH="/opt/conda/envs/GTNM/bin:$PATH"

WORKDIR /workspace
