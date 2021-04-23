import xml.etree.ElementTree as ET
from misc.voc_xml_parser import VocParser
import imageio
import os
import misc.F as util
from imgaug.augmentables.bbs import BoundingBoxesOnImage


class ImEnhance:
    """

    """

    def __init__(self):
        pass

    def augument(self, xml, seq, args, gen_number=1):
        image = imageio.imread(os.path.join(args.img_path, xml.filename))
        bboxes = util.get_BoundingBoxes(xml.bndboxs)  # 取得框框
        bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)  # 畫框在圖上
        # return  image_aug, bbs_aug
        return seq(image=image, bounding_boxes=bbs)

    def __len__(self):
        return len(self.xml_path)


    def __getitem__(self, idx):
        pass


if __name__ == "__main__":
    # xml 的 labeling data
    xml_dlist = VocParser(args.xml_path).get_dlist()

    enhancer = ImEnhance()