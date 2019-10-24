import os
# import json
# import ujson as json
import msgpack
import rapidjson as json
import gzip
import numpy as np
import gc
from enum import IntEnum


class Image(IntEnum):
    ID = 0
    WIDTH = 1
    HEIGHT = 2


class Annotation(IntEnum):
    ID = 0
    IMAGE_ID = 1
    NUM_KEYPOINTS = 2
    ISCROWD = 3


class SlowArray:
    def __init__(self, cols, capacity=0):
        self.data = np.empty((0, cols), np.int64)

    def update(self, row):
        self.data = np.vstack((self.data, row))

    def finalize(self):
        return self.data


class FastArray:
    def __init__(self, cols, capacity=100):
        self.data = np.empty((capacity, cols), np.int64)
        self.capacity = capacity
        self.size = 0
        self.cols = cols

    def update(self, row):
        self.add(row)

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = np.empty((self.capacity, self.cols), np.int64)
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data


class coco(object):
    def __init__(self, filename, fast=True, loadonly=False):
        _, ext = os.path.splitext(filename)
        self.filename = filename
        self.fast = True
        self.genarray = FastArray if fast else SlowArray

        if ext == '.gz':
            self.__open = self.__open_gzip
        elif ext == '.msgpack':
            self.__open = self.__open_msgpack
        else:
            self.__open = self.__open_raw_json
        if not loadonly:
            self.__parse()

    def __open_gzip(self):
        with gzip.open(self.filename) as f:
            data = json.load(f)
        return data

    def __open_raw_json(self):
        with open(self.filename, buffering=2**10) as f:
            data = json.load(f)
        return data

    def __open_msgpack(self):
        with open(self.filename, 'rb') as f:
            data = msgpack.unpack(f, raw=False)
        return data

    def __parse(self):
        data = self.__open()
        np_image_keys = ['id', 'width', 'height']
        np_images = self.genarray(len(np_image_keys), 10000)
        self.images = {}
        for image in data['images']:
            self.images[image['id']] = image
            np_image = [image[x] for x in np_image_keys]
            np_images.update(np_image)
        self.np_images = np_images.finalize()

        np_annotation_keys = ['id', 'image_id', 'num_keypoints', 'iscrowd']
        np_annotations = self.genarray(len(np_annotation_keys), 10000)
        self.annotations = {}
        for annotation in data['annotations']:
            self.annotations[annotation['id']] = annotation
            np_annotation = [annotation[x] for x in np_annotation_keys]
            np_annotations.update(np_annotation)
        self.np_annotations = np_annotations.finalize()
        self.data = data

    def get_image_count(self):
        return len(self.images)

    def get_annotation_count(self):
        return len(self.annotations)

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        del self.data
        del self.images
        del self.np_images
        del self.annotations
        del self.np_annotations
        gc.collect()
