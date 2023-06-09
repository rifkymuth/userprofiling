# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
# FROM ubuntu:18.04
# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN apt-get update

# RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN conda install -c conda-forge hdbscan

RUN pip install --no-cache-dir -r requirements.txt --no-deps hdbscan

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

# sh
# gcloud builds submit --tag gcr.io/imposing-kayak-389216/index
# gcloud run deploy --image gcr.io/impoding-kayak-389216/index --platform managed