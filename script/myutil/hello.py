#!/usr/bin/env python

import sys
import rospy
from sensor_msgs.msg import Image
import cv2
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
# subprocess.run(args=["source", "/home/$USER/cv_bridge_ws/install/setup.zsh"])
from cv_bridge import CvBridge


class ABCNode:
    """
    ROSノードの基本とするクラス（画像）

    以下のような手順を実行します．
    1. 画像をサブスクライブ
    2. 画像を処理
    3. 処理した画像をパブリッシュ
    """

    def __init__(
            self,
            sub_topic:str='/sub/image',
            pub_topic:str='/pub/image'
        ):

        self.sub_img  = rospy.Subscriber(sub_topic, Image, self.update_cvimage)
        self.pub_img  = rospy.Publisher(pub_topic, Image, queue_size=1)
        self.bridge   = CvBridge()


    def update_cvimage(self, msg):
        """
        画像をサブスクライブする関数
        サブスクライバが呼ばれた際に，self.cv_imageにOpenCVの画像を代入します

        Args:
            msg: メッセージ（rosサブスクライバにコールバックされる際に入力されます）
        
        Returns:
            void
        
        Examples:
            subImage = rospy.Subscriber('/xxx/image', Image, imageCallback)
        """

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
        except Exception as e:
            rospy.logerr(e)


    def publish_cvimage(
            self,
            imshow:bool=False,
            windowname:str='image'
        ):
        """
        画像を処理してパブリッシュする関数

        Args:
            void
        
        Returns:
            void
        
        Examples:
            # In main function.
            while not rospy.is_shutdown():
                n.publish_cvimage() # n is instance
                r.sleep() # r is ros.Rate()
        """

        self.p_image = self.image_processing_()
        self.pub_img.publish(
            self.bridge.cv2_to_imgmsg(self.p_image, 'bgr8')
        )
        
        if (imshow is True):
            cv2.imshow(windowname, self.p_image)
            cv2.waitKey(3)


    def image_processing(self):
        """
        画像処理を行う関数
        （オーバーライドして下さい）

        Examples:
            self.cv_imageを画像処理
            処理した画像を返します．
        """

        return self.cv_image