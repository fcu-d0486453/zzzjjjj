from imgaug.augmentables.bbs import BoundingBox
import numpy as np
import os

def get_BoundingBoxes(bboxeslist):
    assert np.array(bboxeslist).ndim == 2

    res = []
    for bbox in bboxeslist:
        # x1,x2, y1,y2 ,  VOC Pascal is (x1,y1,x2,y2)

        x1, x2, y1, y2 = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])
        res.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
    return res


def ensure_folder(folder_path, exists_remake = False):
    '''
    確保某個資料夾必定存在，因為會重新建立。
    '''
    if os.path.isdir(folder_path):
        if not exists_remake:
            return
        else:
            pass # TODO: 刪除既有資料夾 and remake.

    else:
        pass # TODO: 新建該資料夾



if __name__ == "__main__":
    res = get_BoundingBoxes([[10, 20, 60, 70]])

    print("exit")