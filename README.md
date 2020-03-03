pull container 19.02 / 19.06   py2 
 
``` shell
docker pull nvcr.io/nvidia/tensorrt:19.02-py2
docker pull nvcr.io/nvidia/tensorrt:19.06-py2
```


## run container 

--privileged --runtime=nvidia 

(19.02 for yolo2onnx , TRT 5.0.2 ,  onnx 1.3.0)
```shell
docker run --rm -it --privileged --runtime=nvidia -v ~/Desktop/python:/sharefolder nvcr.io/nvidia/tensorrt:19.02-py2 bash
pip install wget onnx==1.3.0 ( 1.2.3 or 1.3.0)
cd tensorrt/samples/python/yolov3_onnx/ && python yolov3_to_onnx.py
cp yolov3.onnx /sharefolder/
```

(19.06 for onnx2trt , TRT 5.1.5 , onnx 1.5.0)
```shell
docker run --rm -it --privileged --runtime=nvidia -v ~/Desktop/python:/sharefolder nvcr.io/nvidia/tensorrt:19.06-py2 bash
pip install wget onnx
git clone https://github.com/royinx/trt_yolov3.git
cp /sharefolder/yolov3.onnx trt_yolov3/ && cd trt_yolov3
python onnx_to_tensorrt.py --build --vram 8 --max_batch_size 64 -p fp16
```

-------------------------------------------------


## others 

#### Debug core dump (inside container)

commit new image and run at a new container 

install gdb  `apt-get update && apt-get -y install gdb python2.7-dbg`


$ cd <DIR>
$ gdb python

(gdb) run XXX.py
(gdb) generate-core-file
(gdb) py-bt
(gdb) q

