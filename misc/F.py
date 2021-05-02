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
                  labels_format=['xml'], train_dir='train', valid_dir='vaild', args=None):
    # train
    train_subset_dir = os.path.join(dir, train_dir)
    ensure_folder(train_subset_dir, remake=True)
    train_image_dir = os.path.join(train_subset_dir, "images")
    train_labels_dir =os.path.join(train_subset_dir, "labels")
    ensure_folder(train_image_dir)
    ensure_folder(train_labels_dir)
    # vaild
    valid_subset_dir = os.path.join(dir, valid_dir)
    ensure_folder(valid_subset_dir, remake=True)
    valid_image_dir = os.path.join(valid_subset_dir, "images")
    valid_labels_dir = os.path.join(valid_subset_dir, "labels")
    ensure_folder(valid_image_dir)
    ensure_folder(valid_labels_dir)

    # noi = len(glob(os.path.join(dir, '*.*')))  # 取得所有檔案
    flatten = lambda t: [item for sublist in t for item in sublist]

    all_format = flatten([image_format, labels_format])
    # 篩選副檔名為 format的 檔名。
    fname = [glob(os.path.join(dir, f'*.{fm}')) for fm in all_format]

    # 蒐集不同檔名的名稱，只取名稱。
    all_avali_file = list(set([item.rsplit('.')[0] for item in flatten(fname)]))
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

    # train image
    for fn in train_sub:
        for sub_n in image_format:
            target_f = fn+f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=train_image_dir)
    print("已將 train image子集 檔案複製至 {}".format(train_image_dir))
    # train label
    for fn in train_sub:
        for sub_n in labels_format:
            target_f = fn+f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=train_labels_dir)
    print("已將 train label子集 檔案複製至 {}".format(train_labels_dir))

    # valid image
    for fn in valid_sub:
        for sub_n in image_format:
            target_f = fn + f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=valid_image_dir)
    print("已將 valid image子集 檔案複製至 {}".format(valid_image_dir))
    # valid label
    for fn in valid_sub:
        for sub_n in labels_format:
            target_f = fn + f'.{sub_n}'
            if os.path.isfile(target_f):
                shutil.copy2(src=target_f, dst=valid_labels_dir)
    print("已將 valid label子集 檔案複製至 {}".format(valid_labels_dir))


def get_image_filenames(dir, format=['png', 'jpeg', 'jpg'], full_path=True):
    flatten = lambda dl: [e for sub in dl for e in sub]
    res = flatten([glob(os.path.join(dir, f'*.{fm}')) for fm in format])
    if full_path:
        return res
    else:
        return [_.replace('\\', ' ').replace('/', ' ').replace('.', ' ').split(' ')[-2] for _ in res]


def img_folder_chk(dir, format=['png', 'jpeg', 'jpg'], logger=None):
    noi = len(glob(os.path.join(dir, '*.*')))
    fname = [glob(os.path.join(dir, f'*.{fm}')) for fm in format ]

    for imtype in fname:
        for img_p in imtype:
            Image.open(img_p).convert('RGB').save(img_p)
    if logger is not None:
        logger.info("image folder check successfully!")


def show_bbox_on_image(bboxs, img_path, mode='xyxy', save=True):

    image = imageio.imread(img_path)

    xyxy = []
    if mode.lower()=='xyxy':
        for x0, y0, x1, y1 in bboxs:
            xyxy.append(BoundingBox(int(x0), int(y0), int(x1), int(y1)))
    elif mode.lower()=='xywh':
        w, h = image.shape[1::-1]
        for xc, yc, bw, bh in bboxs:
            xmin = float(xc) * w - (w/2) * float(bw)
            ymin = float(yc) * h - (h/2) * float(bh)
            xmax = float(xc) * w + (w/2) * float(bw)
            ymax = float(yc) * h + (h/2) * float(bh)
            xyxy.append(BoundingBox(int(xmin), int(ymin), int(xmax), int(ymax)))
    else:
        raise ValueError(f'Not support mode "{mode}". Consider use xyxy(voc format) or xywh(yolo format).')
    bbs = BoundingBoxesOnImage(xyxy, shape=image.shape)

    bbs.draw_on_image(image, size=2)
    imgaug.imshow(bbs.draw_on_image(image, size=2))
    if save:
        imageio.imsave(os.path.join(os.getcwd(), 'test_show.jpg'), bbs.draw_on_image(image, size=2))


def show_bbox_on_image_xmlver(xml_path, img_path, save=False):

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

    # 處理節點範例如下
    <object>
		<name>QRCode</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>352</xmin>
			<ymin>321</ymin>
			<xmax>478</xmax>
			<ymax>501</ymax>
		</bndbox>
	</object>
	"""
    # DEPENDENT!!! string
    def get_value_object_node(xmin, ymin, xmax, ymax):
        n_object = ET.Element('object')
        name = ET.SubElement(n_object, 'name')
        pose = ET.SubElement(n_object, 'pose')
        truncated = ET.SubElement(n_object, 'truncated')
        difficult = ET.SubElement(n_object, 'difficult')
        bndbox = ET.SubElement(n_object, 'bndbox')
        xmin_n = ET.SubElement(bndbox, 'xmin')
        ymin_n = ET.SubElement(bndbox, 'ymin')
        xmax_n = ET.SubElement(bndbox, 'xmax')
        ymax_n = ET.SubElement(bndbox, 'ymax')

        name.text = 'QRCode'
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'
        xmin_n.text = '0'#str(xmin)
        ymin_n.text = '0'#str(ymin)
        xmax_n.text = '0'#str(xmax)
        ymax_n.text = '0'#str(ymax)
        return n_object

    ensure_folder(rewrite_dir)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for object_node in root.findall('object'):
        root.remove(object_node)

    for xmin, ymin, xmax, ymax in xyxy:
        nn = get_value_object_node(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        root.append(nn)  # 在 最後一個 node 後面添加新的節點

    tree.write(os.path.join(rewrite_dir, rewrite_fname))

def Deprecated_rewrite_xyxy2xml(xyxy, xml_path, rewrite_dir, rewrite_fname, label=['xmin', 'ymin', 'xmax', 'ymax']):
    """
    注意這目前只適用於單一物件的偵測使用。多目標的他不會管誰是誰就直接取代。

    將某一個xml的所有 object 的 bbox 都帶換掉

    # xyxy: default: ['xmin', 'ymin', 'xmax', 'ymax']。

    # label: !!!DEPENDENT on !!! xml label format.
    """
    ensure_folder(rewrite_dir)
    tree = ET.parse(xml_path)
    root = tree.getroot()



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

    show_bbox_on_image('')


