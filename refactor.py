from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

from common import *
from turbojpeg import TurboJPEG
from line_profiler import LineProfiler

jpeg = TurboJPEG()

profile = LineProfiler()

class TensorRT(object):
# class TensorRT(RootClass):
    def __init__(self,engine_file):
        super().__init__()
        self.TRT_LOGGER = trt.Logger()
        self.engine = self.get_engine(engine_file)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = self.engine.max_batch_size
        self.allocate_buffers()

    def get_engine(self, engine_file_path):
        if os.path.exists(engine_file_path):
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, \
                trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            print("Cannot find .trt engine file.")
            exit(0)

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
    @profile
    def inference(self,inputs:np.ndarray) -> list: # input: <NCHW>
        self.inputs[0].host = inputs
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        self.context.execute_async(batch_size=len(inputs), 
                                   bindings=self.bindings, 
                                   stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        self.stream.synchronize()
        return [out.host for out in self.outputs]

class YoloTRT(object):
    def __init__(self):
        super().__init__()
        # resolution
        self.preprocessor = PreprocessYOLO((608, 608))
        self.trt = TensorRT("yolov3.trt")
        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],                    
                          "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), 
                                           (59, 119), (116, 90), (156, 198), (373, 326)],
                          "obj_threshold": 0.6,
                          "nms_threshold": 0.5,
                          "yolo_input_resolution": (608, 608)}
        self.postprocessor = PostprocessYOLO(**postprocessor_args)
        
    # def _preprocess(self, input_array:np.ndarray) -> np.ndarray: # 
    #     return self.preprocessor.process(input_array) # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>

    def _inference(self, input: np.ndarray) -> list: # 
        trt_outputs = self.trt.inference(input)
        output_shapes = [(self.trt.max_batch_size, 255, 19, 19), 
                         (self.trt.max_batch_size, 255, 38, 38), 
                         (self.trt.max_batch_size, 255, 76, 76)]
                         
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]  # [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
        return trt_outputs # in: <NCHW> <N,3,608,608>, out: [(N, 255, 19, 19), (N, 255, 38, 38), (N, 255, 76, 76)]

    # def _postprocess(self, feat_batch, shape_orig_WH:tuple): 
    #     return [[self.postprocessor.process(feat,shape_orig)]for feat, shape_orig in zip(feat_batch,shape_orig_WH)]

    @profile
    def inference(self, input_array:np.ndarray): # img_array <N,H,W,C>

        pre = self.preprocessor.process(input_array) # in: <NHWC> raw image batch , out: <NCHW> resized <N,3,608,608>
        trt_outputs = self._inference(pre) # out: [(N, 255, 19, 19), (N, 255, 38, 38), (N, 255, 76, 76)]

        feat_batch = [[trt_outputs[j][i] for j in range(len(trt_outputs))] for i in range(len(trt_outputs[0]))]
        post = [[self.postprocessor.process(feat,input_array.shape)]for feat in feat_batch] # out:[[bbox,score,categories,confidences],...]
        post = post[:len(input_array)]
        return post

def unit_test():
    input_image_path = 'debug_image/two_face.jpg'
    
    with open(input_image_path, 'rb') as infile:
        image_raw = jpeg.decode(infile.read())
        image_raw = image_raw[:,:,[2,1,0]]
    image_raw = np.tile(image_raw,[400,1,1,1])

    yolo = YoloTRT()
    batch_size = yolo.trt.max_batch_size
    for i in range(0, len(image_raw), batch_size):
        batch = image_raw[i:i+batch_size]
        rs = yolo.inference(batch)
        # print(len(rs),type(rs))
    # print(len(rs[0][0]),type(rs[0][0]))
    # print(rs[0][0])
    profile.print_stats()

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
