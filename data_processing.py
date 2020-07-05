import math
from PIL import Image
import numpy as np
from scipy.special import expit
import cv2

# YOLOv3-608 has been trained with these 80 categories from COCO:
# Lin, Tsung-Yi, et al. "Microsoft COCO: Common Objects in Context."
# European Conference on Computer Vision. Springer, Cham, 2014.

def load_label_categories(label_file_path):
    categories = [line.rstrip('\n') for line in open(label_file_path)]
    return categories

LABEL_FILE_PATH = 'coco_labels.txt'
ALL_CATEGORIES = load_label_categories(LABEL_FILE_PATH)

# Let's make sure that there are 80 classes, as expected for the COCO data set:
CATEGORY_NUM = len(ALL_CATEGORIES)
assert CATEGORY_NUM == 80


class PreprocessYOLO(object):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for YOLOv3-608.
    """

    def __init__(self, yolo_input_resolution):
        """Initialize with the input resolution for YOLOv3, which will stay fixed in this sample.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution

    def process(self, input_array:np.ndarray) -> np.ndarray: # <NHWC>
        #input img = 4d NHWC
        image_array = self._resize(input_array)
        image_array = self._shuffle_and_normalize(image_array)
        return image_array

    def _resize(self, input_array:np.ndarray) -> np.ndarray: # <NHWC>
        """Load an image from the specified path and resize it to the input resolution.
        Return the input image before resizing as a PIL Image (required for visualization),
        and the resized image as a NumPy float array.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        output_array = np.empty((input_array.shape[0],*self.yolo_input_resolution,3), dtype=np.float32)
        for n, image in enumerate(input_array):
            output_array[n] = cv2.resize(image, self.yolo_input_resolution, interpolation=cv2.INTER_LINEAR)
        output_array = np.ascontiguousarray(output_array)

        return output_array #<NHWC>

    def _shuffle_and_normalize(self, input_array:np.ndarray):
        """Normalize a NumPy array representing an image to the range [0, 1], and
        convert it from HWC format ("channels last") to NCHW format ("channels first"
        with leading batch dimension).

        Keyword arguments:
        image -- image as three-dimensional NumPy float array, in HWC format
        """
        np.divide(input_array,255.0, out = input_array)
        # NHWC to NCHW format:
        input_array = np.transpose(input_array, [0, 3, 1, 2])
        # Convert the image to row-major order, also known as "C order":
        input_array = np.ascontiguousarray(input_array)
        return input_array


class PostprocessYOLO(object):
    """Class for post-processing the three outputs tensors from YOLOv3-608."""

    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 obj_threshold,
                 nms_threshold,
                 yolo_input_resolution):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.masks = yolo_masks
        self.anchors = yolo_anchors
        self.object_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.input_resolution_yolo = yolo_input_resolution

    def process(self, outputs, resolution_raw):
        """Take the YOLOv3 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input image in NHWC order
        """
        outputs_reshaped = list()
        outputs = [output.reshape(1,*output.shape) for output in outputs]
        for output in outputs:
            # print(output.shape)
            outputs_reshaped.append(self._reshape_output(output))

        preds = self._process_yolo_output(outputs_reshaped, resolution_raw)
        preds = self.reformatting(preds,resolution_raw)
        # 0-3: boxes, 4: categories, 5: confidences

        return preds

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + CATEGORY_NUM)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _process_yolo_output(self, outputs_reshaped, resolution_raw):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """

        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)
            
        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        _, height, width, _ = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None
            # return None, None, None

        boxes = np.concatenate(nms_boxes).astype(int)
        categories = np.concatenate(nms_categories)
        categories = np.expand_dims(categories,axis = -1)
        confidences = np.concatenate(nscores)
        confidences = np.expand_dims(confidences,axis = -1)
        
        preds = np.concatenate((boxes,categories,confidences),axis=1)
        return preds

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        # output_reshaped = np.ascontiguousarray(output_reshaped)
        box_xy = expit(output_reshaped[..., :2])
        
        box_wh = np.exp(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = expit(output_reshaped[..., 4])

        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = expit(output_reshaped[..., 5:6]) # for only people detection
        #box_class_probs = expit(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        # box_xy += grid
        np.add(box_xy,grid,out=box_xy)
        # box_xy /= (grid_w, grid_h)
        np.divide(box_xy,(grid_w, grid_h),out=box_xy)
        # box_wh /= self.input_resolution_yolo
        np.divide(box_wh,self.input_resolution_yolo, out = box_wh)
        # box_xy -= (box_wh / 2.)
        np.subtract(box_xy,(box_wh / 2.), out = box_xy)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: object confidence level, box_class_probs: class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)

        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    def reformatting(self, preds: np.ndarray, shape: tuple) -> np.ndarray: 
        """
        Box (x, y, width, height) to (x1, y1, x2, y2)
        preds: [(x1,x2,y1,y2,class,cls_score), ...]
        shape: ori image shape / (N,H,W,C)
        out: array with x,y coordinate in axis=1  / (N_obj, 4) = [(x1,x2,y1,y2,class,cls_score), ...]
        """
        _, image_raw_height, image_raw_width, _ = shape
        out = []
        if preds is not None:
            bboxes, class_, class_score  = preds[:,0:4],preds[:,4],preds[:,5]
            for idx, bbox in enumerate(bboxes):
                # handle the  of padding space
                x_coord, y_coord, width, height = bbox
                x1 = max(0, np.floor(x_coord + 0.5).astype(int))
                y1 = max(0, np.floor(y_coord + 0.5).astype(int))
                x2 = min(image_raw_width, np.floor(x_coord + width + 0.5).astype(int))
                y2 = min(image_raw_height, np.floor(y_coord + height + 0.5).astype(int))

                # handle the edge case of padding space
                x1 = min(image_raw_width, x1)
                x2 = min(image_raw_width, x2)
                if x1 == x2:
                    continue
                y1 = min(image_raw_height, y1)
                y2 = min(image_raw_height, y2)
                if y1 == y2:
                    continue
                if abs(x2-x1)<=10 | abs(y2-y1)<=10:
                    continue
                out.append([x1,y1,x2,y2,class_[idx],class_score[idx]])
        else:
            out.append(None)

        return np.array(out)