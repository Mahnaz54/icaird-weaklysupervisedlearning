FROM nvcr.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# pass these into build using
# --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg GID=$(data writer group id)
# you can get the group id for data-writer using:
# getent group david | awk -F: '{printf "%s\n", $3}'
# using 1337 as a default so that it can be seen easily if used
# note - these are reset to the current user by the entry point
# script.
ARG UID=1002
ARG GID=1003
ARG DATA_WRITER_GROUP_ID=1005

LABEL maintainer="iCAIRD"

# scripts updates to base image
# install sudo and gosu tools
RUN \
  apt-get update -y \
  && apt-get install -y \
  && apt-get install -y build-essential \
  && apt-get -y install sudo gosu

# Add user ubuntu with no password, add to sudo group
RUN groupadd -g $GID -o ubuntu
RUN adduser --uid $UID --gid $GID --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
RUN chmod a+rwx /home/ubuntu/

# Install Anaconda
WORKDIR "/tmp"
RUN \
  sudo apt-get install -y curl \
  && curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh \
  && bash Anaconda3-2019.10-Linux-x86_64.sh -b \
  && rm Anaconda3-2019.10-Linux-x86_64.sh

ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN \
  conda update conda \
  && conda update anaconda \
  && conda update --all

# mount cwd to the project dir in the container
# allow the ubuntu user to own and thus write to it
ADD --chown=ubuntu . /home/ubuntu/icaird-weaklysupervisedlearning
WORKDIR "/home/ubuntu/icaird-weaklysupervisedlearning"

# set up the wsi-learning project
SHELL ["/bin/bash", "-c"]
RUN make create_environment
RUN conda init bash
RUN echo "conda activate icaird-weaklysupervisedlearning" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN make requirements

# we are going to log in as root and then run the setup script
USER root
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]
