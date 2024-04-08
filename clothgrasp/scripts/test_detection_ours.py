#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
from datetime import datetime
from cv_bridge import CvBridge
from utils.detection_visualizer import DetectionVisualizer
from detectedge_ours import EdgeDetector
import yaml

with open("../config/config.yaml", "r") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

class ClothGrasper:
    def __init__(self):
        self.base_path = param['base_save_path']
        self.detection_method = param['detection_method']
        self._init_vis()

        self.bridge = CvBridge()

    def _init_vis(self):
        self.crop_dims = param[
            'crop_dims'] if self.detection_method == 'network' or self.detection_method == 'groundtruth' else param[
            'crop_dims_baselines']
        self.visualizer = DetectionVisualizer(self.detection_method, self.crop_dims)

    # def _call_detectedge_service(self):
    #     rospy.wait_for_service('detect_edges')
    #     detect_edge = rospy.ServiceProxy('detect_edges', DetectEdge)
    #     return detect_edge()

    def _aggregate_data(self, rgb_im, depth_im, prediction, oe, ie, cor):
        corners = None
        outer_edges = None
        inner_edges = None

        detection_method = param['detection_method']
        if detection_method != self.detection_method:
            self.detection_method = detection_method
            self._init_vis()

        if self.detection_method == 'groundtruth':
            # impred = self.bridge.imgmsg_to_cv2(prediction)
            impred = prediction
        if self.detection_method == 'network':
            corners = cor
            outer_edges = oe
            inner_edges = ie
            impred = np.zeros((corners.shape[0], corners.shape[1], 3), dtype=np.uint8)
            impred[:, :, 0] += corners
            impred[:, :, 1] += outer_edges
            impred[:, :, 2] += inner_edges
        elif self.detection_method == 'clothseg':
            impred = prediction
        elif self.detection_method == 'canny' or self.detection_method == 'canny_color':
            detection = prediction
            impred = detection[:, :, 0]
        elif self.detection_method == 'depthgrad':
            grads = prediction
            impred = grads[:, :, 0]
        elif self.detection_method == 'harris' or self.detection_method == 'harris_color':
            detection = prediction
            impred = detection[:, :, 0]
        return {
            'rgb_im': rgb_im,
            'depth_im': depth_im,
            'prediction': prediction,
            'image_pred': impred,
            'corners': corners,
            'outer_edges': outer_edges,
            'inner_edges': inner_edges,
        }

    def _save(self, data, plot):
        tstamp = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        dir_path = os.path.join(self.base_path, tstamp)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        np.save(os.path.join(dir_path, 'depth.npy'), data['depth_im'])
        rgb_im = cv2.cvtColor(data['rgb_im'], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, "rgb.png"), rgb_im)
        np.save(os.path.join(dir_path, 'pred.npy'), data['prediction'])

        if self.detection_method == 'network':
            impred = cv2.cvtColor(data['image_pred'], cv2.COLOR_BGR2RGB)
        else:
            impred = data['image_pred']
        cv2.imwrite(os.path.join(dir_path, 'impred.png'), impred)

        plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(dir_path, 'plot.png'), plot)
        np.save(os.path.join(dir_path, 'data.npy'), data)

    def run(self):
        # detectedge_response = self._call_detectedge_service()
        ed = EdgeDetector()
        rgb_im, depth_im, prediction, outer_edges, inner_edges, corners = ed.detect_edge()
        data = self._aggregate_data(rgb_im, depth_im, prediction, outer_edges, inner_edges, corners)
        plot = self.visualizer.visualize(data, show_grasp=False)
        return data, plot


if __name__ == '__main__':
    cgs = ClothGrasper()
    data, plot = cgs.run()
    cgs._save(data, plot)
