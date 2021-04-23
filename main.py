# -*- coding: utf-8 -*-
import argparse
from misc.voc_xml_parser import VocParser
import torch
import imgaug
import imageio
import matplotlib.pyplot as plt
import os
import misc.F as util
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug.augmentables.bbs import BoundingBox
import imgaug.augmenters as iaa
from  imageEnhance import ImEnhance

parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', type=str, default=r'.\data\label-qr-code', help='VOC格式標記檔的資料夾')
parser.add_argument('--img_path', type=str, default=r'.\data\raw_qr', help='原始QRCODE的資料夾')
parser.add_argument('--aug_folder', type=str, default=r'.\data\augumented', help='放置被強化過後的資料夾')

args = parser.parse_args()




if __name__ == "__main__":
    # xml 的 labeling data
    xml_dlist = VocParser(args.xml_path).get_dlist()

    enhancer = ImEnhance()


    # enhence 的 seq
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 0.7)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    for xml in xml_dlist:

        image_aug, bbs_aug = enhancer.augument(xml, seq, args)

        image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
        imgaug.imshow(image_after)
        break


    print("exit")