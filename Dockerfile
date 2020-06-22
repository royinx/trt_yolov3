

#Build stage
FROM nvcr.io/nvidia/tensorrt:19.02-py2 AS build-env
RUN pip install wget onnx==1.3.0 && \
    cd tensorrt/samples/python/yolov3_onnx/ && \
    python yolov3_to_onnx.py

#Output Stage
FROM nvcr.io/nvidia/tensorrt:19.12-py3
COPY . .
COPY --from=build-env /workspace/tensorrt/samples/python/yolov3_onnx/yolov3.onnx .


# python onnx_to_tensorrt.py --build --vram 8 --max_batch_size 64 -p fp16
