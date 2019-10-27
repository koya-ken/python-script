import coco
from coco import Annotation,Image
import argparse
import collections
# import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-i', dest='inputfile', type=str, required=True)
    # http://ja.pymotw.com/2/argparse/
    parser.add_argument("inputfile", action="store")
    return parser.parse_args()


def main():
    args = getargs()
    with coco.coco(args.inputfile) as data:
        print('image count:', len(data.np_images[:, Image.ID]))
        print('annotation count:', len(data.np_annotations[:, Annotation.ID]))
        print('image ids', data.np_images[:, Image.ID])
        print('annotation ids', data.np_annotations[:, Annotation.ID])


if __name__ == "__main__":
    main()
