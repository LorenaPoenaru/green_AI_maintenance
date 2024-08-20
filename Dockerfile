FROM python:3.8

# Set bash as the default shell
ENV SHELL=/bin/bash

# Create a working directory
WORKDIR /usr/src/app

# Build with some basic utilities
RUN apt-get update && apt-get install -y \
    python3-pip \
    apt-utils \
    vim \
    git

# alias python='python3'
RUN ln -s /usr/bin/python3 /usr/bin/python

# build with some basic python packages
RUN pip install \
    numpy \
    scipy \
    codecarbon \
    jupyterlab \
    scikit-learn \
    seaborn \
    tqdm

COPY *.py .

ENTRYPOINT ["python"]
CMD [""]