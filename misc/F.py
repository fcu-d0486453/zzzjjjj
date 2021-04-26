from imgaug.augmentables.bbs import BoundingBox
import numpy as np
import os
import shutil


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
            os.makedirs(folder_path, 0x755)
            if logger is not None:
                logger.info("已重新建立 {} 資料夾".format(folder_path))
    else:
        os.makedirs(folder_path, 0x755)
        if logger is not None:
            logger.info("已經建立 {} 資料夾".format(folder_path))


if __name__ == "__main__":
    import logger

    Logger = logger.Logger('./')