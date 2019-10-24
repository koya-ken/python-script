import coco
import numpy as np
from coco import Image,Annotation
import collections
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', dest='inputfile', type=str, required=True)
parser.add_argument('-o', dest='outputfile', default=None, type=str)

args = parser.parse_args()

# data = coco.coco('annotations/person_keypoints_train2017.json')
# data = coco.coco('person_keypoints_train2017.msgpack')
data = coco.coco(args.inputfile)


print(data.np_images[0])
print(data.np_annotations)
id_max = np.max(data.np_annotations[:,Annotation.ID])

a = np.where(data.np_annotations[:,Annotation.ID] < 183830)
print(data.np_annotations[a])
print(data.np_annotations.shape)
print(data.get_image_count())
print(data.get_annotation_count())

annotation_imageids = data.np_annotations[:,Annotation.IMAGE_ID]
c = collections.Counter(annotation_imageids)
a = np.array(c.most_common(),np.int64)
print(a)
print(np.count_nonzero(a[:,1] == np.min(a[:,1])))

if args.outputfile is not None:
    np.savetxt(args.outputfile, data.np_annotations[:,Annotation.ID], fmt='%s')
# np.savetxt("annotation_ids.txt", data.np_annotations[:,Annotation.ID])
