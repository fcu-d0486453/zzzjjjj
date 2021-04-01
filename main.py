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



parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', type=str, default=r'D:\ppppppppppp\!Zhang-Jia Project\program\label-qr-code', help='Voc xml folder path')
parser.add_argument('--img_path', type=str, default=r'C:\Users\mslab\Desktop\git-repo\zjpj\data\raw_qr', help='Raw image folder path')
args = parser.parse_args()


if __name__ == "__main__":
    # xml 的 labeling data
    xml_dlist = VocParser(args.xml_path).get_dlist()

    for xml in xml_dlist:


        image = imageio.imread(os.path.join(args.img_path, xml['filename']))

        bboxes = util.get_BoundingBox_list(xml['bndboxs'])

        bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)

        # BoundingBox(x1=image.shape[1]*0.2,
        #                                                 x2=image.shape[1]*0.6,
        #                                                 y1=image.shape[0]*0.5,
        #                                                 y2=image.shape[0]*0.66)

        imgaug.imshow(bbs.draw_on_image(image, size=2))

        break


    print("exit")