#!/home/envy/.pyenv/versions/anaconda3-2021.05/envs/segformer_env/bin/python
'''#!/usr/bin/python3.6'''
import rospy
import time
from myutil.hello_segformer import SegformerNode


def main(
        config:str='./local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py',
        checkpoint:str='./checkpoints/fch_yugo_4_scapes_b0_iter_28000.pth',
        imshow:bool=False
    ):
    rospy.init_node('SegFormerNode')

    timer = rospy.Rate(10)

    node = SegformerNode(
        config=config,
        checkpoint=checkpoint,
        sub_topic='/omniStereo_node/image0',
        pub_topic='/segformer/segmentation_map'
    ) 

    time.sleep(1)

    i_ = 0
    while not rospy.is_shutdown():
        print(f'\r({i_}/?)', end="")

        node.publish_cvimage(imshow=imshow, windowname='segmented image')

        i_ += 1
        timer.sleep()


if __name__ == '__main__':
    main()