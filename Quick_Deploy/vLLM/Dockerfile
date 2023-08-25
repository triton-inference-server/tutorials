# Currently, vllm supports only cuda 10.8. The latest Triton container release with cuda 10.8 is 22.12. 
FROM nvcr.io/nvidia/tritonserver:22.12-py3

# install python dependencies
RUN pip install --upgrade pip setuptools

# install rest of python required packages
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

CMD ["tritonserver", "--help"]

