FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.1-cpu-py36-ubuntu18.04
# FROM ubuntu:18.04

# Install necessary dependencies for MMS and SageMaker Inference Toolkit
RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless

RUN apt-get install -y build-essential

RUN apt-get install -y git

RUN apt-get install -y python3-pip

RUN python3 -m pip install --upgrade pip

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip3 install sagemaker-inference multi-model-server retrying

# Copy entrypoint script to the image
RUN mkdir -p /home/model-server/

COPY inference_handler.py /home/model-server/inference_handler.py
COPY handler_service.py /home/model-server/handler_service.py
COPY requirements.txt /home/model-server/requirements.txt

COPY model.pth /home/model-server/model.pth

COPY entrypoint.py /home/model-server/entrypoint.py
RUN chmod +x /home/model-server/entrypoint.py

RUN pip3 install -r /home/model-server/requirements.txt

# Define an entrypoint script for the docker image
ENTRYPOINT ["python3", "/home/model-server/entrypoint.py"]