# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser()

if __name__ == "__main__":

    parser.add_argument('--some_setting', type=str, default='value', help='some desc')
    args = parser.parse_args()

    