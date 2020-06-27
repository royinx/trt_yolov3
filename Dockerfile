FROM nvcr.io/nvidia/tensorrt:19.12-py3

# Install Python
RUN apt-get update && \
    apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender1 \
    libturbojpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip
RUN python3 -m pip install --no-cache-dir -U setuptools

COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
WORKDIR /workshop
COPY . .


