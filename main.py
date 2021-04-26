# -*- coding: utf-8 -*-
import argparse
from misc.voc_xml_parser import VocParser
from misc import logger
import torch
import imgaug
import imageio
import matplotlib.pyplot as plt
import os
import misc.F as F
from misc.F import ensure_folder
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.bbs import BoundingBox
import imgaug.augmenters as iaa
from imageEnhance import ImEnhance
import misc.logger as logger
from random import randint, uniform


logger = logger.Logger(level=logger.logging_INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', type=str, default=r'.\data\label-qr-code', help='VOC格式標記檔的資料夾')
parser.add_argument('--img_path', type=str, default=r'.\data\raw_qr', help='原始QRCODE的資料夾')
parser.add_argument('--aug_folder', type=str, default=r'.\data\augumented', help='放置被強化過後的資料夾')
parser.add_argument('--number', type=int, default=3, help="將一張圖強化幾次")

args = parser.parse_args()

ensure_folder(args.aug_folder)

logger.info("================= args =================")
for k, v in args.__dict__.items():
    logger.info('{}: {}'.format(k, v))
logger.info("========================================")

if __name__ == "__main__":
    test1 = True
    test2 = False

    if test1:
        # xml 的 labeling data
        xml_dlist = VocParser(args.xml_path).get_dlist()

        enhancer = ImEnhance()

        # 只用其中 2 個
        aug = iaa.SomeOf(2, [
            iaa.Affine(rotate=45),
            iaa.AdditiveGaussianNoise(scale=0.2 * 255),
            iaa.Add(50, per_channel=True),
            iaa.Sharpen(alpha=0.5)
        ])

        # enhence 的 seq
        def get_seq():
            return iaa.Sequential([
                iaa.Affine(rotate=randint(-5, 5)),  # 隨機旋轉 A~B
                iaa.Affine(shear=(-16, 16)),  # Shear, 剪力
                iaa.Affine(translate_percent={"x": -0.20}, mode=imgaug.ALL, cval=(0, 255)),
                iaa.Multiply((0.5, 1.5)),  # 對整張圖隨機乘 A~B 之間的值
                iaa.MultiplyElementwise((0.5, 1.5)),  # element-wise 隨機乘 A~B 之間的值
                iaa.ReplaceElementwise(0.1, [0, 255]),  # 椒鹽雜訊
                iaa.Cutout(nb_iterations=(1, 5), size=0.1, squared=False),  # 隨機 cutout，
                iaa.Dropout(p=(0, 0.1)),  # 對 0<= p <= 0.1 數量的像素版分比，將pixel設定成黑色
                iaa.JpegCompression(compression=(70, 99)),  # randomly and uniformly sampled per image
                iaa.Affine(
                    translate_px={"x": randint(-5, 5), "y": randint(-5, 5)},  # 平行移動
                    scale=(uniform(.95, 1), uniform(.95, 1)),  # 分別針對 xy 軸做拉伸
                )
            ])

        for xml in xml_dlist:
            for idx in range(args.number):
                image_aug, bbs_aug = enhancer.augument(xml, get_seq(), args)

                aug_xyxy = []
                # 處理強化後的 bbs_aug
                for i in bbs_aug.items:
                    aug_xyxy.append([i.x1_int, i.y1_int, i.x2_int, i.y2_int])

                aug_xml = "{0}_aug_{1}.xml".format(xml.purefname, str(idx))
                _xml_path = os.path.join(args.xml_path, xml.purefname)+'.xml'
                F.rewrite_xyxy(aug_xyxy, _xml_path, logger.get_log_dir(), aug_xml)

                image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
                imgaug.imshow(image_after)

                aug_img = "{0}_aug_{1}.jpg".format(xml.purefname, str(idx))
                imageio.imsave(os.path.join(logger.get_log_dir(), aug_img), image_aug)

            # ---- end of one image.
            logger.info("{} 強化完畢!".format(xml.filename))
            break


        print("exit")

    if test2:
        pass