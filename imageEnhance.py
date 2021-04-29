import xml.etree.ElementTree as ET
from misc.voc_xml_parser import VocParser
import imageio
import os
import misc.F as util
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import imgaug.augmenters as iaa
from random import randint, uniform
import imgaug

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

    # enhance 的 seq
    enhance_event = [
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
    ]

    @staticmethod
    def get_seq(random_order=False, random_pick=False, pick=2, event=enhance_event, image_ref_size=None):
        """ 取得強化用的 sequence。
        Args:
            random_order: sequential 使用隨機順序。(預設=否)

            random_pick: 是否要隨機(預設=否)

            pick: 當 random=True 時有用。為隨機取樣的數量

            event: 為 list，sequence的種類。

            image_ref_size: 該圖片大小，預設不使用，目的是有規則的設定 translate_px。

        return:
            強化用的 sequence

        assert:
            pick 高於可選的 event 總數。


        """
        if not random_pick:
            if random_order:
                return iaa.Sequential(event, random_order=random_order)
            else:
                return iaa.Sequential(event)
        else:
            assert len(event) >= pick
            return iaa.SomeOf(pick, event)


if __name__ == "__main__":
    # xml 的 labeling data
    xml_dlist = VocParser(args.xml_path).get_dlist()

    enhancer = ImEnhance()