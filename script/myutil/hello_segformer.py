#!/usr/bin/env python
import sys
import cv2
import numpy as np
import rospy

sys.path.append('./empowering_modules')
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv

from myutil.hello import ABCNode


class SegformerNode(ABCNode):
    """
    SegFormerのプログラムのROSラッパ

    MMSegmentationに含まれるSegFormerのプログラムを基にしてラッパプログラムを作成しました．
    このため，MMSegmentationに含まれる多くのセマンティックセグメンテーションモデルもかなり似たプログラムで実行できると思います．
    """

    def __init__(
            self,
            config:str='./local_configs/segformer/B0/segformer.b0.512x1024.city.160k.py',
            checkpoint:str='./checkpoints/segformer.b0.512x1024.city.160k.pth',
            #sub_topic:str='/cut/image',
            sub_topic:str='/unity/omni60_image',
            pub_topic:str='/segformer/segmentation_map'
        ):
        """
        SegformerNode classのインストラクタ

        Args:
            checkpoint: Segformerの重みを選択します
            sub_topic: サブスクライブするトピック名です
            pub_topic: パブリッシュするトピック名です

        Return:
            void

        Examples:
            node = SegformerNode()
        """

        super(SegformerNode, self).__init__(
            sub_topic=sub_topic,
            pub_topic=pub_topic
        )
 
        self.config = config
        self.checkpoint = checkpoint
        self.model = init_segmentor(self.config, self.checkpoint, device='cuda:0')
        rospy.loginfo("*** Init segformer node.")


    def image_processing_(self):
        """
        画像変換をセマンティックセグメンテーションする関数
        ABCNode.image_processing_()のオーバーライド

        Args:
            void

        Returns:
            void
        
        Examples:
            while not rospy.is_shutdown():
                n.publication(imshow=True) # n is instance
                r.sleep() # r is ros.Rate()
        """

        image = self.cv_image
        image = cv2.resize(image, (256, 256)) # <--- Add resizing

        result = inference_segmentor(self.model, image)

        if hasattr(self.model, 'module'):
            self.model = self.model.module
        img = self.model.show_result(image, result, palette=get_palette('cityscapes'), show=False)
        # img = self.model.show_result(image, result, palette=get_palette('ade20k'), show=False)

        palette = get_palette('cityscapes')
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        
        palette = np.array(palette)
        assert palette.shape[0] == 19
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        
        # convert to BGR
        # color_seg = color_seg[..., ::-1]
        self.cv_segmentation_map = mmcv.bgr2rgb(color_seg)

        #img_orig = cv2.resize(img_orig, (1024, 512))
        self.cv_segmentation_map = cv2.resize(self.cv_segmentation_map, (1024, 512)) # add m2.

        # セグメンテーション画像のパブリッシュ
        # self.image_publisher.publish(self.bridge.cv2_to_imgmsg(self.cv_segmentation_map, "bgr8"))
        return self.cv_segmentation_map