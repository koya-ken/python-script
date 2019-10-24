import msgpack
import rapidjson as json

with open('annotations/person_keypoints_train2017.json') as f:
    data = json.load(f)

with open('data.msgpack', 'wb') as outfile:
    msgpack.pack(data, outfile, use_bin_type=False)
