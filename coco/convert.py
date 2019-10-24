import msgpack
import rapidjson as json
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', dest='inputfile', type=str, required=True)
parser.add_argument('-o', dest='outputfile', type=str, required=True)

args = parser.parse_args()

with open(args.inputfile) as f:
    data = json.load(f)

with open(args.outputfile, 'wb') as outfile:
    msgpack.pack(data, outfile, use_bin_type=False)
