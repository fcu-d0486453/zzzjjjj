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
import misc.logger as loggerr
from random import randint, uniform
from tqdm import tqdm

logger = loggerr.Logger(level=logger.logging_INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', type=str, default=r'./data/label-qr-code', help='VOC格式標記檔的資料夾')
parser.add_argument('--img_path', type=str, default=r'./data/raw_qr', help='原始QRCODE的資料夾')
parser.add_argument('--aug_folder', type=str, default=r'./data/augumented', help='放置被強化過後的資料夾')
parser.add_argument('--number', type=int, default=3, help="將一張圖強化幾次")
parser.add_argument('--channel-check', action='store_true', help="當 img_path 內的圖片有可能出現alpha通道時，會先處理該folder內的所有圖片。")
args = parser.parse_args()

if args.channel_check:
    F.img_folder_chk(dir=args.img_path, logger=logger)

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

        pbar = tqdm(xml_dlist)
        for xml in pbar:
            pbar.set_description("處理 {}".format(xml.filename))
            pbar.set_postfix({"xml路徑": xml.xml_path})
            for idx in tqdm(range(args.number), leave=False):  # 單張圖片的強化數量

                # 取得對應xml於的image。
                seq = enhancer.get_seq()
                image_aug, bbs_aug = enhancer.augument(xml, seq, args)
                print(seq.get())
                aug_xyxy = []
                # 處理強化後的 bbs_aug
                for i in bbs_aug.items:
                    aug_xyxy.append([i.x1_int, i.y1_int, i.x2_int, i.y2_int])

                aug_fname = "{0}_aug_{1}".format(xml.purefname, str(idx))
                _xml_path = os.path.join(args.xml_path, xml.purefname)+'.xml'
                F.rewrite_xyxy2xml(aug_xyxy, _xml_path, logger.get_log_dir(), f'{aug_fname}.xml')

                image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
                imgaug.imshow(image_after)

                imageio.imsave(os.path.join(logger.get_log_dir(), f'{aug_fname}.jpg'),
                               image_aug)

            # ---- end of one image.
            logger.info("{} 強化完畢!".format(xml.filename))
            # break

        print("exit")

    if test2:
        pass