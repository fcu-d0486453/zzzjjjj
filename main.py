# -*- coding: utf-8 -*-
import argparse
from misc.voc_xml_parser import VocParser
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--xml_path', type=str, default=r'D:\ppppppppppp\!Zhang-Jia Project\program\label-qr-code', help='Voc xml folder')
args = parser.parse_args()


if __name__ == "__main__":
    # xml 的 labeling data
    xml_dlist = VocParser(args.xml_path).get_dlist()



    print("exit")