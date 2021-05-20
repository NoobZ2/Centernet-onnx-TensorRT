# -*-coding:utf-8-*-
import time

import cv2
import numpy as np
import tensorrt as trt
import torch
import pickle
from decode import ctdet_decode, ctdet_post_process
from image import get_affine_transform
#from .detection import DetectionObj
from trt_function import do_inference, allocate_buffers
from trt_inference import TrtInference, Singleton
from scipy.spatial import distance as dist
from utils.debugger import Debugger

@Singleton
class CenterNetTrtInference(TrtInference):

    def __init__(self):
        #super().__init__()
        TRT_LOGGER = trt.Logger()

        self.mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100

        #self.opt = opts
        self.net_inp_height, self.net_inp_width = 512,512
        self.net_out_height, self.net_out_width = self.net_inp_height // 4, self.net_inp_width // 4
        self.classes_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
        self.num_classes = len(self.classes_names)

        self.m_obj_engine = self.load_engine()
        self.detection_inputs, self.detection_outputs, self.detection_bindings, self.detection_stream = allocate_buffers(
            self.m_obj_engine)
        self.detection_context = self.m_obj_engine.create_execution_context()
        self.pause = True

    @property
    def __meta_c(self):
        return np.array([self.image_width / 2., self.image_height / 2.], dtype=np.float32)

    @property
    def __meta_s(self):
        return max(self.image_height, self.image_width) * 1.0

    def load_engine(self):
        engine_path = "ctdet_coco_dlav0_1x.trt"
        with open(engine_path, "rb") as obj_f, trt.Runtime(TrtInference.TRT_LOGGER) as runtime:
            obj_engine = runtime.deserialize_cuda_engine(obj_f.read())
        return obj_engine

    def pre_process(self, image):
        self.image_height, self.image_width = image.shape[0:2]
        trans_input = get_affine_transform(self.__meta_c, self.__meta_s, 0, [self.net_inp_width, self.net_inp_height])
        inp_image = cv2.warpAffine(image, trans_input, (self.net_inp_width, self.net_inp_height),
                                   flags=cv2.INTER_LINEAR)

        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(1, 3, self.net_inp_height, self.net_inp_width)

        img_numpy = np.ascontiguousarray(images, dtype=np.float32)
        return img_numpy

    def infer(self, image, pre_result=None):
        infer_time = time.time()
        image = self.pre_process(image)
        if pre_result:
            output = pickle.load(open(pre_result, "rb"))
        else:
            output = self.trt_infer(image)
            if hasattr(self.opt, "trt_net_res") and self.opt.trt_net_res:
                pickle.dump(output, open("{}/{:.6f}.obj".format(self.opt.infer_path, infer_time), "wb"))

        results = self.__bbox_decode(output)
        objlist = self.__convert(results, infer_time)
        reslist = self.__filter_same_bbox(objlist)
        return reslist

    def convert(self, results, infer_time):
        objlist = []
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                bbox_dict = dict()
                score = bbox[4]
                if score > 0.1:
                    categrey_id = j - 1
                    bbox_dict['timestamp'] = infer_time
                    bbox_dict["class_name"] = self.opt.classes_names[categrey_id]
                    bbox_dict["class_id"] = categrey_id
                    bbox_dict["topleft_x"] = max(0, bbox[0])
                    bbox_dict["topleft_y"] = max(0, bbox[1])
                    bbox_dict["bottomright_x"] = min(self.image_width, bbox[2])
                    bbox_dict["bottomright_y"] = min(self.image_height, bbox[3])
                    bbox_dict["score"] = score
                    obj = DetectionObj(**bbox_dict)
                    objlist.append(obj)
        return objlist

    def __filter_same_bbox(self, obj_list):
        res_list = []
        if len(obj_list) == 0:
            return res_list

        match_dict = {}
        obj_list_centroids = [
            (obj.topleft_x + (obj.bottomright_x - obj.topleft_x) / 2,
             obj.topleft_y + (obj.bottomright_y - obj.topleft_y) / 2) for obj in obj_list
        ]
        obj_distance = dist.cdist(np.array(obj_list_centroids), np.array(obj_list_centroids))
        min_dis_index = np.argsort(obj_distance)
        for i in range(len(obj_list)):
            if min_dis_index[i][0] in match_dict:
                match_dict[min_dis_index[i][0]].append(i)
            else:
                match_dict[min_dis_index[i][0]] = [i]
        for k, v in match_dict.items():
            if len(v) == 1 and k == v[0]:
                res_list.append(obj_list[k])
            else:
                score_list = []
                for i in v:
                    score_list.append(obj_list[i].score)
                max_index = score_list.index(max(score_list))
                last_obj_index = v[max_index]
                res_list.append(obj_list[last_obj_index])
        return res_list

    def trt_infer(self, image):
        self.detection_inputs[0].host = image

        output = do_inference(
            context=self.detection_context,
            bindings=self.detection_bindings,
            inputs=self.detection_inputs,
            outputs=self.detection_outputs,
            stream=self.detection_stream)

        return output

    def bbox_decode(self, infer_out):
        hm = torch.from_numpy(infer_out[0]).sigmoid_()
        wh = torch.from_numpy(infer_out[1])
        reg = torch.from_numpy(infer_out[2])

        hm = hm.reshape(1, self.num_classes, self.net_out_height, self.net_out_width)
        wh = wh.reshape(1, 2, self.net_out_height, self.net_out_width)
        reg = reg.reshape(1, 2, self.net_out_height, self.net_out_width)

        dets = ctdet_decode(hm, wh, reg=reg, K=100)
        print(dets)
        dets = self.__box_post_process(dets)
        results = self.__merge_outputs([dets])
        return results

    def __box_post_process(self, dets):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(dets.copy(), [self.__meta_c], [self.__meta_s], self.net_out_height,
                                  self.net_out_width,
                                  self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= 1
        return dets[0]

    def __merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections], axis=0).astype(np.float32)
            # soft_nms(results[j], Nt=0.5, method=2)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])

        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > 0.3:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)



if __name__ == '__main__':
    cntrt=CenterNetTrtInference()
    debugger = Debugger(dataset='coco', ipynb=(3),
                        theme='white')
    print(type(cntrt))
    #cntrt=cntrt.__init__()
    img=cv2.imread("/home/jiqing/yolov5/data/images/bus.jpg")
    img_numpy=cntrt.pre_process(img)
    output=cntrt.trt_infer(img_numpy)
    results = cntrt.bbox_decode(output)
    #print(output)
    print("results",type(results))
    print(results)
    cntrt.show_results(debugger,img,results)
   # objlist = cntrt.convert(results, time.time())
    #print(img_numpy.shape)
    #print(cntrt.m_obj_engine)