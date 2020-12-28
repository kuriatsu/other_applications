#!/usr/bin/python
# -*- coding: utf-8 -*-
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import tensorflow
import tensorflow.compat.v1 as tensorflow
from functools import partial
from collections import deque
import logging

import rospy
import tf
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped

from vector_map_msgs.msg import PointArray
from vector_map_msgs.msg import LineArray
from vector_map_msgs.msg import StopLineArray
from autoware_msgs.msg import Signals


import art
logging.basicConfig(level=logging.ERROR)

def model(placeholder, initial_model):
    """construct infer model
    ~args~
    placeholder: image placeholder (40x20 image)
    trained_model: model with trained coefficients
    ~return~
    result: list of probability [black, green, red, yellow]
    """
    my_batch_norm_layer = partial(tensorflow.layers.batch_normalization, training=initial_model, momentum=0.9)
    conv1 = tensorflow.layers.conv2d(placeholder, filters=32, kernel_size=3, strides=1, padding="SAME", name="conv1")
    bn1 = tensorflow.nn.relu(my_batch_norm_layer(conv1))
    conv2 = tensorflow.layers.conv2d(bn1, filters=64, kernel_size=3, strides=2, padding="SAME", name="conv2")
    bn2 = tensorflow.nn.relu(my_batch_norm_layer(conv2))
    pool3 = tensorflow.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tensorflow.reshape(pool3, shape=[-1, 64 * 5 * 10])
    fc1 = tensorflow.layers.dense(pool3_flat, 64, name="fc1")
    bn3 = tensorflow.nn.relu(my_batch_norm_layer(fc1))
    logits = tensorflow.layers.dense(bn3, 4, name="output")
    result = tensorflow.nn.softmax(logits)
    return result


class ExtractRoi():
    """extract tlr image acording to roi info from feat_proj
    feat_proj params should be x=1, y=-2
    """

    def __init__(self):
        self.img = None                 # image_raw (cv_image) (None while extract_start_flag is False)
        self.ego_pose = None            # SSIA
        self.roi_img = None             # extracted tlr image
        self.tlr_id = None              # tlr id

        self.extract_start_dist = 10000 # start get roi image with distance (extact_stact_flog must be always True) 40
        self.extract_stop_dist = 0      # stop get roi image with distance (extact_stact_flog must be always True) 20
        self.extract_start_flag = False # get cv_image and extract roi
        self.roi_frame_size_x = 40      # SSIA
        self.roi_frame_size_y = 20      # SSIA

        self.cv_bridge = CvBridge()     # instance for CvBridge change ros_image to cv_image

        ## pub roi image
        # self.pub_roi = rospy.Publisher('/roi_image', Image, queue_size=5)


    def ndtPoseCb(self, in_pose):
        """ndt callback
        ~return~
        self.ego_pose: pose of ego vehicle
        """

        self.ego_pose = in_pose.pose


    def imgCb(self, in_img):
        """image callback
        ~args~
        in_img: /image_raw
        ~return~
        self.img: cv_image
        """

        # if extract_extract_start_flag is False, skip saving image and keep self.img=None
        if not self.extract_start_flag: return

        try:
            self.img = self.cv_bridge.imgmsg_to_cv2(in_img, 'bgr8')

        except:
            logging.error('could not convert from {} to bgr8'.format(in_img.encoding))


    def roiCb(self, in_roi):
        """roi_signal callback
        ~args~
        in_roi: roi info from /roi_signal
        ~return~
        self.tlr_id: id of red light connecting to stop line with 'Tlid'
        """

        # if the signal is road traffic light with 3 lights
        if len(in_roi.Signals) == 3 and self.ego_pose is not None:

            # distance from tlr to ego_vehicle
            dist = math.sqrt((self.ego_pose.position.x - in_roi.Signals[0].x) ** 2 + (self.ego_pose.position.y - in_roi.Signals[0].y) ** 2)

            # if extract_start_flag is False and ego_vehicle is not in specified range from tlr, skip getting roi image
            if not self.extract_start_flag or (dist < self.extract_stop_dist or dist > self.extract_start_dist):
                logging.debug('extract_start_flag in ExtractRoi() is Flase')
                return

            # activate saving image_raw in imgCb(), then extract roi from next call
            self.extract_start_flag = True

            # self.img is None just after turning extract_start_flag to True, extract roi from next call
            if self.img is None: return

            # extract roi and save it to class variable
            try:
                self.roi_img = self.img[in_roi.Signals[0].v - self.roi_frame_size_y/ 2 : in_roi.Signals[0].v + self.roi_frame_size_y / 2,
                                        in_roi.Signals[0].u - self.roi_frame_size_x / 2 : in_roi.Signals[0].u + self.roi_frame_size_x / 2
                                    ]
            except:
                logging.warning('extract rectangle is out of frame')

            ## pub roi image
            # self.pub_roi.publish(self.cv_bridge.cv2_to_imgmsg(self.roi_img, 'bgr8'))
            self.tlr_id = in_roi.Signals[2].signalId
            self.img = None



class TlrClassification():
    """classification class !!
    1. prepare model()
    1. make instance of this class
    2. set callback() at main() or somewhere else
    3. call initModel("/path/to/model.ckpt")
    from vector_map_server.srv import GetStopLine

    finally. call close()
    """
    def __init__(self):
        self.holder = None    # place holder (get image and hold it)
        self.inference = None # inference instance
        self.sess = None      # tensorflow interactive session
        self.img = None
        tensorflow.disable_v2_behavior()
        self.color = [(0, 0, 0),
        		      (0, 255, 0), # green
        		      (0, 0, 255), # red
                      (0, 200, 255) # yellow
                      ]


    def initModel(self, model_path):
        """init model and prepare tensorflow session
        ~args~
        model_path: relative model file path
        """
        self.holder = tensorflow.placeholder(tensorflow.float32, shape=[None, 20, 40, 3], name="holder")

        # instance inference model with default model
        innocent_model = tensorflow.placeholder_with_default(False, shape=(), name='training')
        self.inference = model(self.holder, innocent_model)

        # start tensorflow session
        self.sess = tensorflow.InteractiveSession()
        self.sess.run(tensorflow.global_variables_initializer())

        # model saver which is held with tensorflow session
        saver = tensorflow.train.Saver()
        saver.restore(self.sess, model_path)


    def imgCb(self, in_img):
        """calback of tlr image
        """
        # change ros image to numpy image
        cv_bridge = CvBridge()
        self.img = cv_bridge.imgmsg_to_cv2(in_img, "bgr8")
        classify(self.img)


    def classify(self, img):
        np_img = np.array(img).astype('float32')/255
        np_img = np_img.reshape(1,20,40,3)

        # get classification result
        logits_infer = self.sess.run(self.inference, feed_dict={self.holder: np_img})

        ## visualization
        img = cv2.resize(img, None, fx = 10, fy = 10)
        cv2.putText(img, str(max(logits_infer[0])), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color[np.argmax(logits_infer)], thickness=3)
        cv2.namedWindow("tlr_result", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("tlr_result", img)
        cv2.waitKey(30)

        return np.argmax(logits_infer)

    def close(self):
        self.sess.close()
        print('bye')


class DetectStopLine():
    """search stopline from vector map
    """

    def __init__(self):
        self.ego_pose = None            # SSIA
        self.vector_map_points = {}     # dict for points  key: id value: coordinate dict {x, y}
        self.vector_map_lines = {}      # dict for lines of vector map key: id value: point id
        self.vector_map_stop_lines = {} # dict for stop lines of vector map key: id value: line id
        self.check_length = 10.0        # if the stopline-ego_vehicle distance is less than this value, consider that ego_vehicle is crossing stopline
        self.check_eps = 0.00001        # error tolerance when checking cross in isSegmentCross()

        self.tlr_id = None
        self.is_in_front_of_stop_line = False
        # self.pub_point = rospy.Publisher('/debug', PointStamped, queue_size=100)


    def ndtPoseCb(self, pose_stamped):
        """ndt_pose callback
        get pose and save as class variable
        then find stop line in front of ego vehicle
        """

        self.ego_pose = pose_stamped.pose
        self.checkStopLine()


    def checkStopLine(self):
        """find stop line in front of ego vehicle
        extend vector of ego_vehicle to self.check_length [m]
        then check that the ego_vector and stopline segment are cross or not
        ~return~
        self.is_in_front_of_stop_line: stopline detected!!
        self.tlr_id: red light id associated with stop line
        """

        # initialize return valiables
        self.is_in_front_of_stop_line = False
        self.tlr_id = None

        # if vector map info is empty, skip evelything
        if not self.vector_map_points or not self.vector_map_lines or not self.vector_map_stop_lines:
            logging.error('waiting vectormap info')
            return

        # get ego vehicle vector and extend it's length to self.check_length
        e = tf.transformations.euler_from_quaternion((self.ego_pose.orientation.x, self.ego_pose.orientation.y, self.ego_pose.orientation.z, self.ego_pose.orientation.w))
        ego_vehicle_vec = [self.check_length * math.cos(e[2]), self.check_length * math.sin(e[2])]
        ego_vehicle_point = [self.ego_pose.position.x, self.ego_pose.position.y]

        # search stop line in front of ego_vehicle (stopline_id -> line_id -> point_id -> coordinate)
        for stop_line in self.vector_map_stop_lines.values():
            line_id = stop_line.get('lid') # stopline id
            point_1_id = self.vector_map_lines.get(line_id).get('bpid') # stopline point id 1
            point_2_id = self.vector_map_lines.get(line_id).get('fpid') # stopline point id 2
            point_1 = [self.vector_map_points.get(point_1_id).get('y'), self.vector_map_points.get(point_1_id).get('x')]  # stopline point coordinate 1
            point_2 = [self.vector_map_points.get(point_2_id).get('y'), self.vector_map_points.get(point_2_id).get('x')]  # stopline point coordinate 2

            stop_line_vec = [point_2[0] - point_1[0], point_2[1] - point_1[1]] # vector of stopline for check crossing below

            # check ego_vector and stopline vector
            if self.isSegmentCross(ego_vehicle_point, ego_vehicle_vec, point_1, stop_line_vec):

                ## visualize stopline point in fromt of ego_vehicle
                # stopline_point = PointStamped()
                # stopline_point.header.stamp = rospy.Time.now()
                # stopline_point.header.frame_id = 'map'
                # stopline_point.point.x = point_1[0]
                # stopline_point.point.y = point_1[1]
                # self.pub_point.publish(stopline_point)

                # stopline_point.point.x = point_2[0]
                # stopline_point.point.y = point_2[1]
                # self.pub_point.publish(stopline_point)

                self.is_in_front_of_stop_line = True
                self.tlr_id = stop_line.get('tlid')



    def isSegmentCross(self, point_1, vec_1, point_2, vec_2):
        """check 2 vectors(segment: vector and length) is cross or not
        ~return~
        cross or not
        """

        vec_pts = [point_2[0] - point_1[0], point_2[1] - point_1[1]]
        cross_prod_v1_v2 = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]

        if cross_prod_v1_v2 == 0.0:
            return False

        cross_prod_pts_v1 = vec_pts[0] * vec_1[1] - vec_pts[1] * vec_1[0]
        cross_prod_pts_v2 = vec_pts[0] * vec_2[1] - vec_pts[1] * vec_2[0]

        vec_1_crosspt = cross_prod_pts_v1 / cross_prod_v1_v2
        vec_2_crosspt = cross_prod_pts_v2 / cross_prod_v1_v2

        if vec_1_crosspt + self.check_eps < 0 or vec_1_crosspt - self.check_eps > 1 or vec_2_crosspt + self.check_eps < 0 or vec_2_crosspt - self.check_eps > 1:
            return False

        logging.debug('crossing')
        return True


    def vectorMapStopLineCb(self, in_stop_lines):
        """get stopline info from /vector_map_info
        ~return~
        self.vector_map_stop_lines: dict of {lineid(line consists of points), tlid(red light id)}
        """
        for stop_line in in_stop_lines.data:
            self.vector_map_stop_lines[stop_line.lid] = {'lid': stop_line.lid, 'tlid': stop_line.tlid}


    def vectorMapLineCb(self, in_lines):
        """get line info from /vector_map_info
        ~return~
        self.vector_map_lines: dict of {bpid(point1), fpid(point2)}
        """
        for line in in_lines.data:
            self.vector_map_lines[line.lid] = {'bpid': line.bpid, 'fpid': line.fpid}


    def vectorMapPointsCb(self, in_points):
        """get point info from /vector_map_info
        ~return~
        self.vector_map_points: dict of {x(y coordinate in map), y(x coordinate in map)}
        """
        for point in in_points.data:
            self.vector_map_points[point.pid] = {'x': point.bx, 'y': point.ly}


class DetectSignalViolation():
    """detect signal violation
    while the result of DetectStopLine.is_in_front_of_stopline is true,
    extract roi and classify it, then totalize the results just acter 'crossing' stopline
    finally if the majority of results is RED, consider that the vehicle ignore tlr.
    if the majority is GREEN or YELLOW, consider that it's safe.
    """

    def __init__(self):

        self.was_in_front_of_stop_line = False        # the vehicle was crossed in last call?
        self.tlr_results = deque()                    # tlr classification results while 'crossing'

        self.tlr_classification = TlrClassification() # SSIA
        self.tlr_classification.initModel('../TrData/model/model.ckpt')

        self.detect_stop_line = DetectStopLine()      # SSIA

        self.extract_roi = ExtractRoi()               # SSIA
        self.extract_roi.extract_start_flag = False   # extract flag shold be True only while 'crossing'

        sub_image     = rospy.Subscriber('/image_raw', Image, self.extract_roi.imgCb)                                            # for extract_roi
        sub_ndt_pose  = rospy.Subscriber('/ndt_pose', PoseStamped, self.ndtPoseCb)                                               # for extract_roi, detect_stop_line
        sub_roi       = rospy.Subscriber('/roi_signal', Signals, self.extract_roi.roiCb)                                         # for extract_roi
        sub_point     = rospy.Subscriber('/vector_map_info/point', PointArray, self.detect_stop_line.vectorMapPointsCb)          # for detect_stop_line
        sub_line      = rospy.Subscriber('/vector_map_info/line', LineArray, self.detect_stop_line.vectorMapLineCb)              # for detect_stop_line
        sub_stop_line = rospy.Subscriber('/vector_map_info/stop_line', StopLineArray, self.detect_stop_line.vectorMapStopLineCb) # detect_stop_line
        self.pub_result    = rospy.Publisher('/error/tlr_ignore', Bool, queue_size=1)

    def ndtPoseCb(self, current_pose):
        """ndt callback
        ~return~
        self.detect_stop_line.ego_pose : to find stop line in front of ego_vehicle
        self.extract_roi.ego_pose : to check distnce from tlr to ego_vehicle
        """

        self.detect_stop_line.ego_pose = current_pose.pose
        self.extract_roi.ego_pose = current_pose.pose

        # check stop line and judge 'crossing'
        self.detect_stop_line.checkStopLine()

        # if 'crossing' extract roi and classify it
        if self.detect_stop_line.is_in_front_of_stop_line:
            self.extract_roi.extract_start_flag = True

            # check roi_image exists and tlr_id from /roi_signal and stopline_detector are same, then classify image and add result list
            if self.extract_roi.roi_img is not None and self.extract_roi.tlr_id == self.detect_stop_line.tlr_id:
                self.tlr_results.append(self.tlr_classification.classify(self.extract_roi.roi_img))
                self.extract_roi.roi_img = None

        # if just after 'crossing', totalize results and judge violation
        elif not self.detect_stop_line.is_in_front_of_stop_line and self.was_in_front_of_stop_line:
            self.judgeViolation()
            # cleaning up for next call
            self.tlr_results.clear()
            self.extract_roi.extract_start_flag = False

        self.was_in_front_of_stop_line = self.detect_stop_line.is_in_front_of_stop_line


    def judgeViolation(self):
        """judge tlr violation
        ~return~
        result of judge
        """

        # for totalize results [black, red, green, yellow]
        tlr_results_total = [0] * 4

        # totalize results
        for i in range(0, min(10, len(self.tlr_results))):
            tlr_results_total[self.tlr_results.pop()] += 1

        logging.debug(tlr_results_total)

        # if crossed while red
        if tlr_results_total.index(max(tlr_results_total)) in [2, 3]:
            text = art.text2art('out')
            print(text)
            self.pub_result.publish(True)

        # if crossed while green and yellow
        elif tlr_results_total.index(max(tlr_results_total)) == 1:
            text = art.text2art('safe')
            print(text)

        # if no tlr
        else:
            text = art.text2art('where is tlr')
            print(text)


def main():

    rospy.init_node('detect_signal_violation_node')
    detect_signal_violation = DetectSignalViolation()

    rospy.spin()

    # clean up tensorlow
    detect_signal_violation.tlr_classification.close()


if __name__=='__main__':
    main()
