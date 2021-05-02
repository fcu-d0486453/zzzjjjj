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
from misc.myparser import YoloLabelReader
from glob import glob

logger = loggerr.Logger(level=logger.logging_INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--label_path', type=str, default=r'./data/label-qr-code', help='標記檔的資料夾')
parser.add_argument('--img_path', type=str, default=r'./data/raw_qr', help='原始QRCODE的資料夾')
parser.add_argument('--number', type=int, default=1, help="將一張圖強化幾次")
parser.add_argument('--channel-check', action='store_true', help="當 img_path 內的圖片有可能出現alpha通道時，會先處理該folder內的所有圖片。")
parser.add_argument('--verbose', action='store_true', help="印出一堆訊息")
parser.add_argument('--show_augment', action='store_true', help="強化的結果印在圖片上")
args = parser.parse_args()

show_augment_dir_name = "show_augment"

# HARDCODE
args.show_augment = True

if args.channel_check:
    F.img_folder_chk(dir=args.img_path, logger=logger)

if args.show_augment:
    ensure_folder(os.path.join(logger.get_log_dir(), show_augment_dir_name))

logger.info("================= args =================")
for k, v in args.__dict__.items():
    logger.info('{}: {}'.format(k, v))
logger.info("========================================")


if __name__ == "__main__":
    test0 = True
    test1 = False
    test2 = False

    if test0:
        label_reader = YoloLabelReader(label_dir=args.label_path, image_dir=args.img_path)
        fn_list = F.get_image_filenames(args.img_path, full_path=False)
        enhancer = ImEnhance()

        pbar = tqdm(fn_list)
        for fn in pbar:  # ['qr_0009', 'qr_0010']:  # fn_list
            for idx in range(args.number):  # 單張圖片的強化數量
                label_x = label_reader[fn]
                (image_aug, bbs_aug), params = enhancer.augument(label_x)
                fn = fn+f'_{idx}'
                pbar.set_description("處理圖片 {} r={}".format(fn, params['rotation']))
                im_path = os.path.join(logger.get_log_dir(), f'{fn}.jpg')  # ..
                if args.show_augment:
                    image_after_aug = bbs_aug.draw_on_image(image_aug, size=5, color=[255, 0, 0])
                    rd = params['rotation']
                    img_aug_path = os.path.join(logger.get_log_dir(), show_augment_dir_name, f'{fn}_r_{rd}.jpg') #..
                    imageio.imsave(img_aug_path, im=image_after_aug)
                F.write_label_and_image2(im_path, image_aug, fn, bbs_aug, logger)



    if test1:
        # labeling data
        annotation_data = VocParser(args.label_path).get_dlist()

        enhancer = ImEnhance()

        pbar = tqdm(xml_dlist)
        for xml in pbar:
            pbar.set_description("處理 {}".format(xml.filename))
            pbar.set_postfix({"xml路徑": xml.xml_path})
            for idx in tqdm(range(args.number), leave=False):  # 單張圖片的強化數量

                # 取得對應xml於的image。
                seq = enhancer.get_seq(random_order=True, random_pick=False, pick=2)
                image_aug, bbs_aug = enhancer.augument(xml, seq, args)
                # 移除超出邊界框，與切平邊界框
                bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
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

        print(f"logging dir: {logger.get_log_dir()}. (success exit!)")

        F.dataset_split(logger.get_log_dir())

    if test2:
        F.dataset_split(logger.get_log_dir())