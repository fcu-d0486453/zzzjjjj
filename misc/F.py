import misc.logger as logger
from imgaug.augmentables.bbs import BoundingBox
import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import imgaug
from PIL import Image
from glob import glob
import os
import random

def dataset_split(dir, train=0.7, valid=0.3, image_format=['png', 'jpeg', 'jpg'],
                  labels_format=['xml'], train_dir='train', vaild_dir='vaild', args=None):
    train_subset_dir = os.path.join(dir, train_dir)
    vaild_subset_dir = os.path.join(dir, vaild_dir)
    ensure_folder(train_subset_dir, remake=True)
    ensure_folder(os.path.join(train_subset_dir, "images"))
    ensure_folder(os.path.join(train_subset_dir, "labels"))
    ensure_folder(vaild_subset_dir, remake=True)
    ensure_folder(os.path.join(vaild_subset_dir, "images"))
    ensure_folder(os.path.join(vaild_subset_dir, "labels"))
    noi = len(glob(os.path.join(dir, '*.*')))  # 取得所有檔案
    flatten = lambda t: [item for sublist in t for item in sublist]

    all_format = flatten([image_format, labels_format])
    fname = [glob(os.path.join(dir, f'*.{fm}')) for fm in all_format]  # 篩選副檔名為 format的 檔名。

    all_avali_file = list(set([item.rsplit('.')[0] for item in flatten(fname)]))  # 蒐集不同檔名的名稱
    none_repeat = len(all_avali_file)

    if args is not None and args.verbose:
        print(f"image 使用副檔名: {image_format}")
        print(f"label 使用副檔名: {labels_format}")
        print(f"不重複檔名: {all_avali_file}")
        print(f"不重複檔名數量: {none_repeat}")

    train_pick = round(none_repeat * train)
    valid_pick = round(none_repeat * valid)
    # 不想管的話就直接註解掉 assert。
    assert train_pick + valid_pick == none_repeat  # 用意: 一箱蘋果隨意分兩包，兩包數量要等同一箱。
    random.shuffle(all_avali_file)
    train_sub = all_avali_file[0:train_pick]
    valid_sub = all_avali_file[train_pick:]
    if args is not None and args.verbose:
        print("train {}筆: {}".format(len(train_sub), train_sub))
        print("valid {}筆: {}".format(len(valid_sub), valid_sub))
    else:
        print("train {}筆".format(len(train_sub)))
        print("valid {}筆".format(len(valid_sub)))
    print("共 {} 筆.".format(len(train_sub)+len(valid_sub)))

    for fn in train_sub:
        for sub_n in image_format:
            target_f = fn+f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=train_subset_dir)
                # print(f"{target_f} --copy2--> {train_subset_dir}")
    print("已將 train image子集 檔案複製至 {}".format(train_subset_dir))
    for fn in train_sub:
        for sub_n in image_format:
            target_f = fn+f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=train_subset_dir)
    print("已將 train label子集 檔案複製至 {}".format(train_subset_dir))
    # TODO: 將 分類的圖片再加以分到 label 與 images 資料夾內。
    for fn in valid_sub:
        for sub_n in labels_format:
            target_f = fn+f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=os.path.join(fn, target_f), dst=vaild_subset_dir)
                # print(f"{target_f} --copy2--> {vaild_subset_dir}")
    print("已將 vaild子集 檔案複製至 {}".format(vaild_subset_dir))

def img_folder_chk(dir, format=['png', 'jpeg', 'jpg'], logger=None):
    noi = len(glob(os.path.join(dir, '*.*')))
    fname = [glob(os.path.join(dir, f'*.{fm}')) for fm in format ]

    for imtype in fname:
        for img_p in imtype:
            Image.open(img_p).convert('RGB').save(img_p)
    if logger is not None:
        logger.info("image folder check successfully!")

def show_bbox_on_image(xml_path, img_path, save=False):

    image = imageio.imread(img_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xyxy = []

    for _ in tree.findall('object'):
        x0 = _.find('bndbox').find('xmin').text
        y0 = _.find('bndbox').find('ymin').text
        x1 = _.find('bndbox').find('xmax').text
        y1 = _.find('bndbox').find('ymax').text
        xyxy.append(BoundingBox(int(x0), int(y0), int(x1), int(y1)))

    bbs = BoundingBoxesOnImage(xyxy, shape=image.shape)

    bbs.draw_on_image(image, size=2)
    imgaug.imshow(bbs.draw_on_image(image, size=2))
    if save:
        imageio.imsave(os.path.join(os.getcwd(), 'test_show.jpg'), bbs.draw_on_image(image, size=2))

def rewrite_xyxy2xml(xyxy, xml_path, rewrite_dir, rewrite_fname, label=['xmin', 'ymin', 'xmax', 'ymax']):
    """
    注意這目前只適用於單一物件的偵測使用。多目標的他不會管誰是誰就直接取代。

    將某一個xml的所有 object 的 bbox 都帶換掉

    # xyxy: default: ['xmin', 'ymin', 'xmax', 'ymax']。

    # label: !!!DEPENDENT on !!! xml label format.
    """
    ensure_folder(rewrite_dir)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    EACH_LABEL_ELEMENT = len(label)
    # TODO: 當數量不一樣時的防呆，xml 出錯 or 程式員給錯數量的 xyxy。 xyxy總數不符合 object標籤數量。

    for label_idx, clabel in enumerate(label):  # 迭代 label參數
        for idx, val in enumerate(root.iter(clabel)):  # 迭代 xml 標籤
            val.text = str(xyxy[idx][label_idx])  # 修改

    tree.write(os.path.join(rewrite_dir, rewrite_fname))


def get_BoundingBoxes(bboxeslist):
    assert np.array(bboxeslist).ndim == 2

    res = []
    for bbox in bboxeslist:
        # x1,x2, y1,y2 ,  VOC Pascal is (x1,y1,x2,y2)

        x1, x2, y1, y2 = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])
        res.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
    return res


def ensure_folder(folder_path, remake=False, logger=None):
    """
    確保某個資料夾必定存在，因為會重新建立。
    """
    if os.path.isdir(folder_path):
        if not remake:
            if logger is not None:
                logger.info("已經存在 {} 不須建立".format(folder_path))
            return
        else:
            if logger is not None:
                logger.warning("已經存在 {}".format(folder_path))
            shutil.rmtree(folder_path)
            if logger is not None:
                logger.warning("已刪除 {} 資料夾".format(folder_path))
            os.makedirs(folder_path, 0o755)
            if logger is not None:
                logger.info("已重新建立 {} 資料夾".format(folder_path))
    else:
        os.makedirs(folder_path, 0o755)
        if logger is not None:
            logger.info("已經建立 {} 資料夾".format(folder_path))


if __name__ == "__main__":

    show_bbox_on_image(r"D:\Git\zjpj\log_XD\20210427_0231\qr_0009_aug_1.xml",
                       r"D:\Git\zjpj\log_XD\20210427_0231\qr_0009_aug_1.jpg", save=True)

