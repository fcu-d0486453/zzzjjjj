import xml.etree.ElementTree as ET
import glob
import os


class VocParser:
    """
    path: xml folder.
    """
    def __init__(self, path):
        assert os.path.exists(path)
        self.paths = glob.glob(os.path.join(path.rstrip(os.path.sep))+os.path.sep+'*.xml')

    def get_dlist(self):
        """
        return: list(), all of each image bbox coordi and other params.
        """
        tokens = []
        for _ in self:
            tokens.append(_)
        return tokens

    def __getitem__(self, idx):
        tree = ET.parse(self.paths[idx])
        res = dict()
        for _ in tree.findall('path'):
            res.update({'path': _.text})

        for _ in tree.findall('filename'):
            res.update({'filename': _.text})

        for _ in tree.findall('size'):
            res.update({'width': _.find('width').text})
            res.update({'height': _.find('height').text})
            res.update({'depth': _.find('depth').text})
        bndboxs = []
        for _ in tree.findall('object'):
            x0 = _.find('bndbox').find('xmin').text
            y0 = _.find('bndbox').find('ymin').text
            x1 = _.find('bndbox').find('xmax').text
            y1 = _.find('bndbox').find('ymax').text
            bndboxs.append([x0, y0, x1, y1])
        res.update({'bndboxs': bndboxs})

        return res


if __name__ == "__main__":
    # 使用方法，此目錄下為 voc 格式的 xml 們。
    path = r'D:\ppppppppppp\!Zhang-Jia Project\program\label-qr-code'

    # dlist = dict list
    xml_dlist = VocParser(path).get_dlist()

    print("==================\n")
    # 透過迭代的方式取得每筆資料的訊息
    for label_dict in xml_dlist[:3]:
        for key, item in label_dict.items():
            print(key, item)
        print("==================")
    print('exit')