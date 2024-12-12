#!/home/envy/.pyenv/versions/anaconda3-2021.05/envs/segformer_env/bin/python
# ----- !/home/envy/.pyenv/versions/anaconda3-2021.05/envs/segformer/bin/python
'''#!/usr/bin/python3.6'''

import rospy
from sensor_msgs.msg import Image
import time
import sys

import cv2
import numpy as np

print(sys.version)

sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

from cv_bridge import CvBridge

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv

def main():
    rospy.init_node('SegFormer_node')

    timersan = rospy.Rate(10)
    node = ROS_Segformer() 

    time.sleep(1)

    rospy.loginfo('--- Start SegFormer!!! ---')
    while not rospy.is_shutdown():

        start = time.time()
        # -----
        node.publication()
        # -----
        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        timersan.sleep()

class ROS_Segformer:
    def __init__(self):
        self.bridge = CvBridge()

        # self.sub_topic_name = '/omniStereo_node/image0'
        # self.sub_topic_name = '/output_image'
        self.sub_topic_name = '/cut/image'
        # self.sub_topic_name = '/dreamvu/pal/persons/get/left'
        self.pub_topic_name = '/segformer/segmentation_map'
        self.image_subscriber = rospy.Subscriber(self.sub_topic_name, Image, self.imageCallback)
        self.image_publisher  = rospy.Publisher(self.pub_topic_name, Image, queue_size=1)
        
        # SegFormer B0
        self.config = '/home/envy/catkin_ws/src/hello_segformer/script/local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py'
        self.checkpoint = '/home/envy/catkin_ws/src/hello_segformer/script/checkpoints/segformer.b0.512x1024.city.160k.pth'

        # SegFormer B1
        # self.config = '/home/envy/catkin_ws/src/hello_segformer/script/local_configs/segformer/B1/segformer.b1.1024x1024.city.160k.py'
        # self.checkpoint = '/home/envy/catkin_ws/src/hello_segformer/script/checkpoints/segformer.b1.1024x1024.city.160k.pth'
        # self.checkpoint = '/home/envy/catkin_ws/src/hello_segformer/script/checkpoints/fch_yugo_4_scapes_b1_iter_8000.pth'

        # SegFormer B3
        # self.config = '/home/envy/catkin_ws/src/hello_segformer/script/local_configs/segformer/B3/segformer.b3.1024x1024.city.160k.py'
        # self.checkpoint = '/home/envy/catkin_ws/src/hello_segformer/script/checkpoints/segformer.b3.1024x1024.city.160k.pth'

        # ABE20K
        # self.config = './local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py'
        # self.checkpoint = './checkpoints/segformer.b1.512x512.ade.160k.pth'

        self.model = init_segmentor(self.config, self.checkpoint, device='cuda:0')

    def imageCallback(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(e)
            rospy.logerr('error : imageCallback()')
    
    def publication(self):
        rospy.loginfo('---')

        image = self.cv_image
        image = cv2.resize(image, (256, 256)) # <--- Add resizing
        cv2.imshow('input image', image)
        
        result = inference_segmentor(self.model, image)


        # show_result_pyplot(self.model, image, result, get_palette('cityscapes'))
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        img = self.model.show_result(image, result, palette=get_palette('cityscapes'), show=False)
        # img = self.model.show_result(image, result, palette=get_palette('ade20k'), show=False)
 
        cv2.imshow('image segmentation', mmcv.bgr2rgb(img))
        # cv2.waitKey(3)

        palette = get_palette('cityscapes')
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        
        palette = np.array(palette)
        # assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[0] == 19
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        self.cv_segmentation_map = mmcv.bgr2rgb(color_seg)
        cv2.imshow('image segmentation (color map)', self.cv_segmentation_map)
        cv2.waitKey(3)

        #img_orig = cv2.resize(img_orig, (1024, 512)) # add m2.
        self.cv_segmentation_map = cv2.resize(self.cv_segmentation_map, (1024, 512)) # add m2.
        self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.cv_segmentation_map, "bgr8"))



        
if __name__ == '__main__':
    main()
