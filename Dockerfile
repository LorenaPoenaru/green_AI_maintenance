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
    numpy=1.25.1 \
    scipy=1.10.1 \
    pandas=2.0.2 \
    matplotlib=3.7.2 \
    codecarbon \
    scikit-learn=1.2.2 \
    seaborn=0.12.2 \
    tqdm=4.65.0 \
    xgboost

COPY *.py .

ENTRYPOINT ["python"]
CMD [""]