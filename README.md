pull container 19.02 py2 / 19.12 py3
 
``` shell
docker pull nvcr.io/nvidia/tensorrt:19.02-py2
docker pull nvcr.io/nvidia/tensorrt:19.12-py3
```

## Run conatainer from 1-Step mini build
```shell
docker build -t trt_yolov3 .
docker run -it --rm --runtime=nvidia -v $PWD:/workshop -w /workshop trt_yolov3 python3 refactor.py --build --vram 6 --max_batch_size 64

```


## Run container from scratch

--privileged --runtime=nvidia 

(19.02 for yolo2onnx , TRT 5.0.2 ,  onnx 1.3.0 or 1.2.3 )
```shell
docker run --rm -it --privileged --runtime=nvidia -v ~/Desktop/python:/sharefolder nvcr.io/nvidia/tensorrt:19.02-py2 bash
pip install wget onnx==1.3.0 
cd tensorrt/samples/python/yolov3_onnx/ && python yolov3_to_onnx.py
cp yolov3.onnx /sharefolder/
```

(19.12 for onnx2trt , TRT 6.0.1.8 , onnx 1.7.0)
```shell
docker run --rm -it --privileged --runtime=nvidia -v ~/Desktop/python:/sharefolder nvcr.io/nvidia/tensorrt:19.12-py3 bash
pip3 install wget onnx scipy line_profiler
git clone https://github.com/royinx/trt_yolov3.git
cp /sharefolder/yolov3.onnx trt_yolov3/ && cd trt_yolov3
python3 refactor.py --build --vram 6 --max_batch_size 64
```

-------------------------------------------------


## Others 

#### Debug core dump (inside container)

commit new image and run at a new container 

install gdb  `apt-get update && apt-get -y install gdb python2.7-dbg`


$ cd <DIR>
$ gdb python

```
(gdb) run XXX.py
(gdb) generate-core-file
(gdb) py-bt
(gdb) q
```
