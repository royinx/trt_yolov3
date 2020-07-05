from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import asyncio

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

from common import *
from turbojpeg import TurboJPEG

import time 

import threading 
from queue import Queue

pre_q = Queue()
inf_q = Queue(30) # avoid out of memory
post_q = Queue()
rs_q = Queue()

jpeg = TurboJPEG()

from time import perf_counter
import cv2
# Memory Profiling
# from memory_profiler import profile

# Single Thread Profiling
# from line_profiler import LineProfiler
# profile = LineProfiler()

# Multiple Thread Profiling
# import yappi



MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE'))
NUM_OF_TRT_ENGINE = int(os.environ.get('NUM_OF_TRT_ENGINE',1))

RUN = 0
tic = None
class GPU_thread(threading.Thread):
# class TensorRT(RootClass):
    def __init__(self,daemon=None):
        super().__init__(daemon=daemon)
        self.cuda_ctx = None  # to be created when run
        self.engine = None   # to be created when run
        self.running = None
        
    def stop(self):
        self.running = False
        self.join()

    # @profile
    def run(self):
        self.running = True
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.engine = YoloGPU("yolov3.trt")
        print('TrtThread: start running...')
        while self.running:
            if RUN: # start inference concurrently
                if not inf_q.empty():
                    input_array, ori_shape = inf_q.get()
                    outputs = self.engine.inference(input_array) # out: [(N, 255, 19, 19), (N, 255, 38, 38), (N, 255, 76, 76)]
                    while 1:
                        if not post_q.full():
                            post_q.put((outputs, ori_shape))
                            del outputs
                            break
                        time.sleep(0.1)
                else:
                    time.sleep(0.05)
                    
                    
        del self.engine
        self.cuda_ctx.pop()
        del self.cuda_ctx

class YoloGPU(object):
# class TensorRT(RootClass):
    def __init__(self,engine_file):
        super().__init__()
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine(engine_file)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = self.engine.max_batch_size
        assert MAX_BATCH_SIZE == self.max_batch_size
        self.allocate_buffers()

    def get_engine(self, engine_file_path):
        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            raise("Cannot find .trt engine file.")

    def allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            # print(bindings)
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        # return inputs, outputs, bindings, stream


    def inference(self,inputs:np.ndarray) -> list: # input: <NCHW>
        self.inputs[0].host = inputs
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async(batch_size=len(inputs), 
                                   bindings=self.bindings, 
                                   stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [out.host for out in self.outputs]

class YoloCPU(object):
    def __init__(self):
        super().__init__()
        # resolution
        self.preprocessor = PreprocessYOLO((608, 608))
        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), 
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": (608, 608)}
        self.postprocessor = PostprocessYOLO(**postprocessor_args)
        self.running = True
        
    def stop(self):
        self.running = False

    def reshape_CUDA_inf(self, trt_outputs):
        output_shapes = [(MAX_BATCH_SIZE, 255, 19, 19), 
                        (MAX_BATCH_SIZE, 255, 38, 38), 
                        (MAX_BATCH_SIZE, 255, 76, 76)]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]  # [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
        return trt_outputs # in: <NCHW> <N,3,608,608>, out: [(N, 255, 19, 19), (N, 255, 38, 38), (N, 255, 76, 76)]

    def preprocess(self):
        while self.running:
            if not pre_q.empty():
                input_array = pre_q.get()
                outputs = self.preprocessor.process(input_array) # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>
                while 1:
                    if not inf_q.full():
                        inf_q.put((outputs, input_array.shape))
                        del outputs
                        break
                    time.sleep(0.1)
            else:
                time.sleep(0.05)


    # @profile
    def postprocess(self): # img_array <N,H,W,C>
        while self.running:
            if not post_q.empty():
                input_array, ori_shape = post_q.get()
                input_array = self.reshape_CUDA_inf(input_array)
                feat_batch = [[input_array[j][i] for j in range(len(input_array))] for i in range(len(input_array[0]))]
                outputs = [self.postprocessor.process(feat,ori_shape)for feat in feat_batch] # out:[[bbox,score,categories,confidences],...]
                outputs = outputs[:ori_shape[0]]

                while 1:
                    if not rs_q.full():
                        rs_q.put(outputs)
                        del outputs
                        break
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

def unit_test():
    yolo_cpu = YoloCPU()
    batch_size = MAX_BATCH_SIZE

    threads = {"cpu_threads":[], "gpu_threads":[]}
    threads["cpu_threads"].append(threading.Thread(target = yolo_cpu.preprocess, daemon = True) )
    threads["cpu_threads"].append(threading.Thread(target = yolo_cpu.postprocess, daemon = True) )
    for _ in range(NUM_OF_TRT_ENGINE):
        threads["gpu_threads"].append(GPU_thread(daemon = True))

    for instance in threads.values(): # go to type instance (CPU/GPU)
        for instance_thread in instance: # go to instance thread (CPU thread / GPU thread)
            instance_thread.start()


    print("wait for pushing image....")
    # time.sleep(3)
    print("start pushing image....")

    input_image_path = 'debug_image/test2.jpg'
    with open(input_image_path, 'rb') as infile:
        image_raw_ = jpeg.decode(infile.read())
        image_raw = image_raw_[:, :, [2,1,0]]
    load_batch_size = 500
    image_raw = np.tile(image_raw,[load_batch_size,1,1,1])

    for i in range(0, len(image_raw), batch_size):
        batch = image_raw[i:i+batch_size]
        pre_q.put(batch)
        del batch
    del image_raw

    time.sleep(5)
    # GPU worker delay timer
    def start_timer():
        # synchronise n-GPU threads to start service
        time.sleep(0)
        print("""================== start infffffff ==================""")
        global RUN,tic
        tic = perf_counter()
        RUN = 1
        return 
    timer = threading.Thread(target = start_timer)
    timer.start()
    
    while 1:
        rs_size = rs_q.qsize()
        print(f'pre: {pre_q.qsize()}, inf: {inf_q.qsize()}, post: {post_q.qsize()}, rs: {rs_size}')
        for _ in range(rs_size):
            out = rs_q.get()
        if pre_q.qsize() == inf_q.qsize() == post_q.qsize() == 0:
            
            yolo_cpu.running = False

            for thread_ in threads["cpu_threads"]:
                thread_.join()
            print("cpu joined")
            
            for idx, gpu_thread in enumerate(threads["gpu_threads"]):
                gpu_thread.stop()
                print(f"GPU_thread:{idx} joined")
            timer.join()
            # profile.print_stats()
            # return
            break
        else:
            time.sleep(0.05)
    total_time = perf_counter()-tic
    throughput = load_batch_size/total_time
    print(f'total: {load_batch_size}, shape: {image_raw_.shape}, batch: {MAX_BATCH_SIZE}, time: {round(total_time,4)}, throuhput: {throughput}')
    # # reformatting result
    
    print(out)
    # for x in out:
    #     print(x)
    preds = out[0]
    image_out = image_raw_
    if preds is not None:
        for pred in preds:
            image_out = cv2.rectangle(image_out, tuple(pred[0:2].astype(int)), tuple(pred[2:4].astype(int)), (255,0,0), 2)
    else:
        print("No pedestrian detected")
    # cv2.imwrite("out2.jpg",image_out)

def build_engine(FLAGS):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    onnx_file_path = FLAGS.onnx
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder,\
            builder.create_network() as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = FLAGS.vram* 1 << 30 # 1GB
        builder.max_batch_size = FLAGS.max_batch_size

        if FLAGS.precision == 'fp16':
            # set to fp16 
            print('force to fp16')
            builder.fp16_mode = True
            builder.strict_type_constraints = True
        elif FLAGS.precision == 'int8':
            # set to int8

            pass
            # builder.int8_mode = True

            '''
            NUM_IMAGES_PER_BATCH = 5 
            batch = ImageBatchStream(NUM_IMAGES_PER_BATCH, calibration_files)
            Int8_calibration = EntropyCalibrator(['input_node_name'],batchstream)
            trt_builder.int8_calibrator = Int8_calibrator
            '''
        else:
            pass

        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        # setting output layer
        # if FLAGS.onnx=="yolov3.onnx":
        #     print(network.num_layers - 1)
        #     output_1 = network.get_layer(82)
        #     output_1 = network.get_layer(-1)
        #     print(output_1)
        #     # output_2 = network.get_layer(94)
        #     # output_3 = network.get_layer(106)
        #     # network.mark_output(output_1.get_output(0),output_2.get_output(0),output_3.get_output(0))
        #     exit()
        # else:
        #     last_layer = network.get_layer(network.num_layers - 1)
        #     network.mark_output(last_layer.get_output(0))


        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        engine_file_path = f'{onnx_file_path.split(".onnx")[0]}_batch_{FLAGS.max_batch_size}.trt'
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        return engine

def main(FLAGS):
    if FLAGS.build:
        build_engine(FLAGS)
    else:
        unit_test()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action="store_true", required=False, default=False,
                        help='Build model')
    parser.add_argument('--vram', type=int, required=False,
                        help='(Build mode) int - Suppose using VRAM size, recommand half of GPU memory.')
    parser.add_argument('--max_batch_size', type=int, required=False,
                        help='(Build mode) Max batch size for memalloc on GPU.')    
    parser.add_argument('-p', '--precision', type=str, choices=['fp32', 'fp16', 'int8'],
                        required=False, default='fp16',
                        help='(Build mode) dtype precision')
    parser.add_argument('--onnx', type=str,
                        required=False, default='yolov3.onnx',
                        help='(Build mode) ONNX model location')

    FLAGS = parser.parse_args()

    main(FLAGS)

    # docker build -t yolotrt . 

    # Unit test
    # docker run -it --rm --runtime=nvidia yolotrt python3 refactor.py 

    # Build engine
    # docker run -it --rm --runtime=nvidia -v ${PWD}:/workshop -w /workshop   yolotrt python3 refactor.py --build --vram 6 --max_batch_size 64
 
    # export NUM_OF_TRT_ENGINE=2 MAX_BATCH_SIZE=36 && clear && clear && python refactor_gputhread.py