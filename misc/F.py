from imgaug.augmentables.bbs import BoundingBox
import numpy as np

def get_BoundingBox_list(bboxeslist):
    assert np.array(bboxeslist).ndim == 2

    res = []
    for bbox in bboxeslist:
        # x1,x2, y1,y2 ,  VOC Pascal is (x1,y1,x2,y2)
        res.append(BoundingBox(x1=bbox[0], x2=bbox[2], y1=bbox[1], y2=bbox[3]))
    return res


if __name__ == "__main__":
    res = get_BoundingBox_list([[10,20,60,70]])

    print("exit")