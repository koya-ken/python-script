import os
# import json
# import ujson as json
import msgpack
import rapidjson as json
import gzip
import numpy as np
import gc
from enum import IntEnum
import copy


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

        self.data[self.size] = np.array(x)
        self.size += 1

    def finalize(self):
        data = self.data[:self.size]
        return data


class CocoId(object):
    def __init__(self, ids):
        self.ids = ids
        pass

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)


class ImageId(CocoId):
    pass


class AnnotationId(CocoId):
    pass


class ReplaceDict(object):
    def __init__(self, item=None):
        self.item = item
        pass

    def __getitem__(self, item):
        if self.item is None:
            return item
        if item in self.item:
            return self.item[item]
        else:
            return item


class coco(object):
    NP_IMAGE_KEYS = ['id', 'width', 'height']
    NP_ANNOTATION_KEYS = ['id', 'image_id', 'num_keypoints', 'iscrowd']

    def __init__(self, filename, fast=True, loadonly=False, fromfile=True, data=None, needcopy=False):
        """コンストラクタ

        Arguments:
            object {object} -- super class
            filename {str} -- datasetのファイル名

        Keyword Arguments:
            fast {bool} -- 生成速度を速くする (default: {True})
            loadonly {bool} -- 読み込みだけ行う (default: {False})
            fromfile {bool} -- ファイルから作成する (default: {True})
            data {dict} -- dictionaryから作成する、fromfileをFalseにして使用する (default: {None})
        """
        if fromfile:
            self.__from_file(filename, fast, loadonly)
        elif data is not None:
            self.__from_data(data)
        self.needcopy = needcopy

    @staticmethod
    def create(data, copydata=False):
        """dictからdatasetを作成する

        Arguments:
            data {dict} -- cocodataset dict

        Keyword Arguments:
            copy {bool} -- 引数のコピーを使用して作成する (default: {False})

        Returns:
            coco -- coco dataset
        """
        if copydata:
            return __class__(None, fromfile=False, data=json.loads(json.dumps(data)))
        return __class__(None, fromfile=False, data=data)

    def get_image_count(self):
        """datasetに含まれるimageの数を返却

        Returns:
            int -- image総数
        """
        return len(self.images)

    def get_annotation_count(self):
        """datasetに含まれるannotationの数を返却

        Returns:
            int -- annotation総数
        """
        return len(self.annotations)

    def update_data(self, np_images, np_annotation):
        """imagesとannotationsを更新

        Arguments:
            np_images {numpy.ndarray} -- numpy images
            np_annotation {numpy.ndarray} -- numpy annotations
        """
        new_images = [self.images[id] for id in np_images[:, Image.ID]]
        self.images = new_images
        new_annotations = [self.annotations[id]
                           for id in np_annotation[:, Annotation.ID]]
        self.annotations = new_annotations
        self.data['images'] = new_images
        self.data['annotations'] = new_annotations
        self.np_images = np_images
        self.np_annotations = np_annotation

    def save(self, outputfile):
        """datasetをjson形式でfileに保存する

        Arguments:
            outputfile {str} -- save path
        """
        with open(outputfile, "w") as f:
            json.dump(self.data, f)

    def get_union_imageid(self, other_ids):
        """datasetに含まれるimageidと引数のdataset/listに含まれるimageidの和集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- imageid set
        """
        image_ids = self.np_images[:, Image.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_images[:, Image.ID]
        else:
            target_ids = list(other_ids)
        return set(np.union1d(image_ids, target_ids).tolist())

    def get_intersection_imageid(self, other_ids):
        """datasetに含まれるimageidと引数のdataset/listに含まれるimageidの積集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- imageid set
        """
        image_ids = self.np_images[:, Image.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_images[:, Image.ID]
        else:
            target_ids = list(other_ids)
        return set(np.intersect1d(image_ids, target_ids).tolist())

    def get_diff_imageid(self, other_ids):
        """datasetに含まれるimageidから引数のdataset/listのidを引いた差集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- imageid set
        """
        image_ids = self.np_images[:, Image.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_images[:, Image.ID]
        else:
            target_ids = list(other_ids)
        return set(np.setdiff1d(image_ids, target_ids).tolist())

    def get_union_annotationid(self, other_ids):
        """datasetに含まれるannotationidと引数のdataset/listに含まれるannotationidの和集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- annotationid set
        """
        annotation_ids = self.np_annotations[:, Annotation.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_annotations[:, Annotation.ID]
        else:
            target_ids = list(other_ids)
        return set(np.union1d(annotation_ids, target_ids).tolist())

    def get_intersection_annotationid(self, other_ids):
        """datasetに含まれるannotationidと引数のdataset/listに含まれるannotationidの積集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- annotationid set
        """
        annotation_ids = self.np_annotations[:, Annotation.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_annotations[:, Annotation.ID]
        else:
            target_ids = list(other_ids)
        return set(np.intersect1d(annotation_ids, target_ids).tolist())

    def get_diff_annotationid(self, other_ids):
        """datasetに含まれるannotationidから引数のdataset/listのidを引いた差集合

        Arguments:
            other_ids {Union[coco,list]} -- image ids

        Returns:
            set -- annotationid set
        """
        annotation_ids = self.np_annotations[:, Annotation.ID]
        if type(other_ids) is coco:
            target_ids = other_ids.np_annotations[:, Annotation.ID]
        else:
            target_ids = list(other_ids)
        return set(np.setdiff1d(annotation_ids, target_ids).tolist())

    def replace_imageid(self, replace_id):
        ids = self.np_images[:, Image.ID]
        id_count = len(ids)
        if type(replace_id) is int:
            rids = range(replace_id, replace_id + id_count)
        elif len(replace_id) == id_count:
            rids = replace_id
        else:
            assert False, 'replace_id is int or id list'

        def swapid(item, newid): return item.update({'id': newid}) or item
        [swapid(item[1], item[0]) for item in zip(rids, self.images.values())]
        self.update_images(self.images)

        replace_id_dict = dict(zip(ids, rids))
        self.update_annotations(self.annotations, replace_id_dict)

        return self

    def replace_annotationid(self, replace_id):
        ids = self.np_annotations[:, Annotation.ID]
        id_count = len(ids)
        if type(replace_id) is int:
            rids = range(replace_id, replace_id + id_count)
        elif len(replace_id) == id_count:
            rids = replace_id
        else:
            assert False, 'replace_id is int or id list'

        def swapid(item, newid): return item.update({'id': newid}) or item
        [swapid(item[1], item[0])
         for item in zip(rids, self.annotations.values())]
        self.update_annotations(self.annotations)

        return self

    def swap_imageid(self, swap_ids):
        old_ids = [id[0] for id in swap_ids]
        i_ids = self.get_intersection_imageid(old_ids)

        assert len(old_ids) == len(i_ids), 'swap_ids not contained.'
        def swapid(item, newid): return item.update({'id': newid}) or item

        [swapid(self.images[i[0]], i[1]) for i in swap_ids]
        self.update_images(self.images)
        self.update_annotations(self.annotations, dict(swap_ids))

    def swap_annotationid(self, swap_ids):
        swap_ids = list(swap_ids)
        old_ids = [id[0] for id in swap_ids]
        i_ids = self.get_intersection_annotationid(old_ids)

        assert len(old_ids) == len(i_ids), 'swap_ids not contained.'
        def swapid(item, newid): return item.update({'id': newid}) or item
        [swapid(self.annotations[i[0]], i[1]) for i in swap_ids]

        self.update_annotations(dict(self.annotations))

    def update_images(self, new_images):
        np_image_keys = self.NP_IMAGE_KEYS
        np_images = self.genarray(len(np_image_keys), 10000)

        self.images = {}
        for image in new_images.values():
            self.images[image['id']] = image
            np_image = [image[x] for x in np_image_keys]
            np_images.update(np_image)
        self.np_images = np_images.finalize()
        self.data['images'] = list(self.images.values())

    def update_annotations(self, new_annotations, swap_image_ids=None):
        np_annotation_keys = self.NP_ANNOTATION_KEYS
        np_annotations = self.genarray(len(np_annotation_keys), 10000)

        self.annotations = {}
        replace_id_dict = ReplaceDict(swap_image_ids)
        for annotation in new_annotations.values():
            old_id = annotation['image_id']
            new_id = replace_id_dict[old_id]
            if new_id not in self.images:
                continue
            self.annotations[annotation['id']] = annotation
            annotation['image_id'] = new_id
            np_annotation = [annotation[x] for x in np_annotation_keys]
            np_annotations.update(np_annotation)
        self.np_annotations = np_annotations.finalize()
        self.data['images'] = list(self.images.values())
        self.data['annotations'] = list(self.annotations.values())

    def filter_imageid(self, ids, includes=True):
        tmp_images = copy.copy(self.images)
        tmp_annotations = copy.copy(self.annotations)
        np_images = copy.copy(self.np_images)
        np_annotations = copy.copy(self.np_annotations)
        i_ids = self.get_intersection_imageid(ids)

        assert len(i_ids) == len(ids), 'ids not contained'

        if not includes:
            ids = self.get_diff_imageid(ids)

        new_images = {}
        for i in ids:
            new_images[i] = self.images[i]

        tmp_data = copy.copy(self.data)
        self.update_images(new_images)
        self.update_annotations(self.annotations)
        ret_data = self.data
        self.images = tmp_images
        self.annotations = tmp_annotations
        self.data = tmp_data
        self.update_data(np_images, np_annotations)
        return coco.create(ret_data, self.needcopy)

    def filter_annotationid(self, ids, includes=True):
        tmp_images = copy.copy(self.images)
        tmp_annotations = copy.copy(self.annotations)
        np_images = copy.copy(self.np_images)
        np_annotations = copy.copy(self.np_annotations)
        i_ids = self.get_intersection_annotationid(ids)

        assert len(i_ids) == len(ids), 'ids not contained'

        if not includes:
            ids = self.get_diff_annotationid(ids)

        new_annotations = {}
        for i in ids:
            new_annotations[i] = self.annotations[i]

        tmp_data = copy.copy(self.data)
        self.update_annotations(new_annotations)

        annotated_ids = self.np_annotations[:, Annotation.IMAGE_ID].tolist()
        annotated_ids = set(annotated_ids)
        ret_coco = self.filter_imageid(annotated_ids)
        self.data = tmp_data
        self.images = tmp_images
        self.annotations = tmp_annotations
        self.update_data(np_images, np_annotations)
        return ret_coco

    def filter(self, exp):
        width = self.np_images[:, [Image.WIDTH]]
        height = self.np_images[:, [Image.HEIGHT]]
        image_id = self.np_annotations[:, [Annotation.IMAGE_ID]]
        iscrowd = self.np_annotations[:, [Annotation.ISCROWD]]
        num_keypoints = self.np_annotations[:, [Annotation.NUM_KEYPOINTS]]

        is_image_exp = False
        is_annotation_exp = False
        for i in self.NP_IMAGE_KEYS:
            is_image_exp |= f'{i} ' in exp
        for i in self.NP_ANNOTATION_KEYS:
            is_annotation_exp |= f'{i} ' in exp

        assert is_image_exp ^ is_annotation_exp, 'invalid filter'

        if is_image_exp:
            matched = self.np_images[np.all(eval(exp), axis=1)][:, Image.ID]
            assert len(matched) > 0, 'no match filter'
            return self & ImageId(matched)
        else:
            matched = self.np_annotations[np.all(
                eval(exp), axis=1)][:, Annotation.ID]
            assert len(matched) > 0, 'no match filter'
            return self & AnnotationId(matched)

    def refresh(self):
        self.__parse(self.data, copydata=self.needcopy)

    def setneedcopy(self, needcopy=True):
        self.needcopy = needcopy

    def __from_data(self, data):
        self.fast = True
        self.genarray = FastArray
        self.__parse(data)

    def __from_file(self, filename, fast=True, loadonly=False):
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
            self.__parse(self.__open())

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

    def __parse(self, data, copydata=False):
        if copydata:
            data = json.loads(json.dumps(data))
        np_image_keys = self.NP_IMAGE_KEYS
        np_images = self.genarray(len(np_image_keys), 10000)
        self.images = {}
        for image in data['images']:
            self.images[image['id']] = image
            np_image = [image[x] for x in np_image_keys]
            np_images.update(np_image)
        self.np_images = np_images.finalize()

        np_annotation_keys = self.NP_ANNOTATION_KEYS
        np_annotations = self.genarray(len(np_annotation_keys), 10000)
        self.annotations = {}
        for annotation in data['annotations']:
            self.annotations[annotation['id']] = annotation
            np_annotation = [annotation[x] for x in np_annotation_keys]
            np_annotations.update(np_annotation)
        self.np_annotations = np_annotations.finalize()
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        del self.data
        del self.images
        del self.np_images
        del self.annotations
        del self.np_annotations
        gc.collect()

    def __merge(self, other, new_annotation_id=False):
        """datasetをマージする

        Arguments:
            other {coco} -- other coco type

        Keyword Arguments:
            new_annotation_id {bool} -- 全てのAnnotationIdを新しく振り直す (default: {False})

        Returns:
            coco -- merged dataset
        """
        assert type(
            other) is self.__class__, f"other is not {self.__class__.__name__} type"

        duplicate_image_ids = self.get_intersection_imageid(other)
        assert len(duplicate_image_ids) == 0, 'duplicate image id'

        new_data = {}
        s_old_ids = self.np_annotations[:, Annotation.ID].tolist()
        o_old_ids = other.np_annotations[:, Annotation.ID].tolist()

        new_data['info'] = self.data['info']
        new_data['licenses'] = self.data['licenses']
        new_data['images'] = list(self.images.values()) + \
            list(other.images.values())
        new_data['annotations'] = None
        new_data['categories'] = self.data['categories']

        duplicate_annotation_ids = self.get_intersection_annotationid(other)
        if len(duplicate_annotation_ids) == 0:
            new_data['annotations'] = list(
                self.annotations.values()) + list(other.annotations.values())
        elif new_annotation_id:
            self.replace_annotationid(0)
            a_count = self.get_annotation_count()
            other.replace_annotationid(a_count)
            new_data['annotations'] = list(
                self.annotations.values()) + list(other.annotations.values())
        else:
            union_annotation_ids = self.get_union_annotationid(other)
            start_id = max(union_annotation_ids) + 1
            end_id = start_id + len(union_annotation_ids)
            dids = duplicate_annotation_ids
            swap_ids = zip(dids, range(start_id, end_id))
            other.swap_annotationid(swap_ids)
            new_data['annotations'] = list(
                dict(self.annotations).values()) + list(dict(other.annotations).values())
            new_data['annotations'] = new_data['annotations']

        s_new_ids = self.np_annotations[:, Annotation.ID].tolist()
        o_new_ids = other.np_annotations[:, Annotation.ID].tolist()
        if self.needcopy:
            self.refresh()
            other.refresh()
            self.swap_annotationid(zip(s_new_ids, s_old_ids))
            other.swap_annotationid(zip(o_new_ids, o_old_ids))

        return new_data

    def __or__(self, other):
        return self.__class__(None, data=self.__merge(other), fromfile=False, needcopy=self.needcopy)

    def __and__(self, other):
        if type(other) is ImageId:
            return self.filter_imageid(other)
        elif type(other) is AnnotationId:
            return self.filter_annotationid(other)
        assert False, 'invalid type'

    def __sub__(self, other):
        if type(other) is ImageId:
            return self.filter_imageid(other, includes=False)
        elif type(other) is AnnotationId:
            return self.filter_annotationid(other, includes=False)
        assert False, 'invalid type'
