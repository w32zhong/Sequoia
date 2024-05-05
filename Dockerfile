FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
ADD ./LLM-common-eval/requirements.txt r1.txt
RUN pip install -r r1.txt
ADD ./sequoia/requirements.txt r2.txt
RUN pip install -r r2.txt
RUN apt update && apt install -y git
# setup the shell
ADD . s3d
WORKDIR /workspace/s3d
CMD /bin/bash
