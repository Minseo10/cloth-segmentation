#!/usr/bin/env python
# import os
# import rospy
import cv2
# import message_filters
import numpy as np
# from clothgrasp.srv import DetectEdge, DetectEdgeResponse
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from copy import deepcopy
from methods.groundtruth import GroundTruth
from methods.clothseg import ClothSegmenter
from methods.model.model import ClothEdgeModel
from methods.canny import CannyEdgeDetector
from methods.canny_color import CannyColorEdgeDetector
from methods.depthgrad import DepthGradDetector
from methods.harris import HarrisDetector
from methods.harris_color import HarrisColorDetector
import yaml

with open("../config/config.yaml", "r") as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

class EdgeDetector():
    """
    Runs service that returns a cloth region segmentation.
    """
    def __init__(self):
        # rospy.init_node('detectedge_service')
        self.bridge = CvBridge()
        self.detection_method = param['detection_method']
        self._init_model()

        self.depth_im = None
        self.rgb_im = None
        # image size: 2208 x 1242 pixels
        self.depth_tiff_path = "../../Dataset/sample_000002/observation_start/depth_map.tiff"
        self.depth_path = "../../Dataset/sample_000002/observation_start/depth_image.jpg"
        self.rgb_path = "../../Dataset/sample_000002/observation_start/image_left.png"
        self.prediction = None
        self.outer_edges = None
        self.inner_edges = None
        self.corners = None
        # self.depthsub = message_filters.Subscriber('/depth_to_rgb/image_raw', Image)
        # self.rgbsub = message_filters.Subscriber('/rgb/image_raw', Image)

        # self.server = rospy.Service('detect_edges', DetectEdge, self._server_cb)

        # self.ts = message_filters.ApproximateTimeSynchronizer([self.depthsub, self.rgbsub], 10, 0.1)
        # self.ts.registerCallback(self._callback)

    def _init_model(self):
        if self.detection_method == 'groundtruth':
            self.crop_dims = param['crop_dims']
            self.model = GroundTruth(self.crop_dims)
        elif self.detection_method == 'network':
            self.crop_dims = param['crop_dims']
            grasp_angle_method = param['grasp_angle_method']
            model_path = param['model_angle_path'] if grasp_angle_method == 'predict' else param['model_path']
            self.model = ClothEdgeModel(self.crop_dims, grasp_angle_method, model_path)
        elif self.detection_method == 'clothseg':
            self.crop_dims = param['crop_dims_baselines']
            D = np.array(param['D'])
            K = np.array(param['K'])
            K = np.reshape(K, (3, 3))
            w2c_pose = np.array(param['w2c_pose'])
            segment_table = param['segment_table']
            table_plane = np.array(param['table_plane'])
            self.model = ClothSegmenter(D, K, w2c_pose, segment_table, table_plane, self.crop_dims)
        elif self.detection_method == 'canny':
            self.crop_dims = param['crop_dims_baselines']
            self.model = CannyEdgeDetector(self.crop_dims)
        elif self.detection_method == 'canny_color':
            self.crop_dims = param['crop_dims_baselines']
            self.model = CannyColorEdgeDetector(self.crop_dims)
        elif self.detection_method == 'depthgrad':
            self.crop_dims = param['crop_dims_baselines']
            self.model = DepthGradDetector(self.crop_dims)
        elif self.detection_method == 'harris':
            self.crop_dims = param['crop_dims_baselines']
            self.model = HarrisDetector(self.crop_dims)
        elif self.detection_method == 'harris_color':
            self.crop_dims = param['crop_dims_baselines']
            self.model = HarrisColorDetector(self.crop_dims)
        else:
            raise NotImplementedError

    def get_image(self, depth_path, rgb_path):
        # depth_im = self.bridge.imgmsg_to_cv2(depth_msg)
        # rgb_im = self.bridge.imgmsg_to_cv2(rgb_msg)
        depth_im = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        rgb_im = cv2.imread(rgb_path)
        # cv2.imshow('depth image', depth_im)
        # cv2.waitKey()
        self.depth_im = np.nan_to_num(depth_im)
        self.rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

    def detect_edge(self):
        # rospy.loginfo('Received cloth detection request')
        self.get_image(self.depth_path, self.rgb_path)

        rgb_im = deepcopy(self.rgb_im)
        depth_im = deepcopy(self.depth_im)
        # if rgb_im is None or depth_im is None:
        #     raise rospy.ServiceException('Missing RGB or Depth Image')

        # response = DetectEdgeResponse()
        # response.rgb_im = self.bridge.cv2_to_imgmsg(rgb_im)
        # response.depth_im = self.bridge.cv2_to_imgmsg(depth_im)

        if self.detection_method == 'groundtruth':
            pred = self.model.predict(rgb_im)
            # response.prediction = self.bridge.cv2_to_imgmsg(pred)
            self.prediction = pred
        elif self.detection_method == 'network':
            self.model.update() # Check if model needs to be reloaded
            #start = rospy.Time.now()
            corners, outer_edges, inner_edges, pred = self.model.predict(depth_im)
            #end = rospy.Time.now()
            #d = end - start
            #rospy.loginfo('Network secs: %d, nsecs: %d' % (d.secs, d.nsecs))

            # response.prediction = self.bridge.cv2_to_imgmsg(pred)
            # response.corners = self.bridge.cv2_to_imgmsg(corners)
            # response.outer_edges = self.bridge.cv2_to_imgmsg(outer_edges)
            # response.inner_edges = self.bridge.cv2_to_imgmsg(inner_edges)
            self.prediction = pred
            self.corners = corners
            self.outer_edges = outer_edges
            self.inner_edges = inner_edges
        elif self.detection_method == 'clothseg':
            mask = self.model.predict(depth_im)
            self.prediction = mask
        elif self.detection_method == 'canny':
            grads = self.model.predict(depth_im)
            self.prediction = grads
        elif self.detection_method == 'canny_color':
            grads = self.model.predict(rgb_im)
            self.prediction = grads
        elif self.detection_method == 'depthgrad':
            grads = self.model.predict(depth_im)
            self.prediction = grads
        elif self.detection_method == 'harris':
            corner_preds = self.model.predict(depth_im)
            self.prediction = corner_preds
        elif self.detection_method == 'harris_color':
            corner_preds = self.model.predict(rgb_im)
            self.prediction = corner_preds

        # rospy.loginfo('Sending cloth detection response')
        return rgb_im, depth_im, self.prediction, self.outer_edges, self.inner_edges, self.corners

if __name__ == '__main__':
    ed = EdgeDetector()
    rgb_im, depth_im, prediction, outer_edges, inner_edges, corners = ed.detect_edge()
    # cv2.imshow('prediction masks', prediction)
    # cv2.waitKey()
    # cv2.imshow('outer edge', outer_edges)
    # cv2.waitKey()
    # cv2.imshow('inner edge', inner_edges)
    # cv2.waitKey()
    # cv2.imshow('corners', corners)
    # cv2.waitKey()
