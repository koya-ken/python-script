import msgpack
import rapidjson as json
import argparse
import os
import gc


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("inputfile", action="store")
    args = parser.parse_args()
    return args


def convert(filepath):
    basefilepath, ext = os.path.splitext(filepath)
    outputfilepath = basefilepath + '.msgpack'
    with open(filepath) as f:
        data = json.load(f)

    with open(outputfilepath, 'wb') as outfile:
        msgpack.pack(data, outfile, use_bin_type=False)
    del data
    gc.collect()


def main():
    args = getargs()
    [convert(file) for file in args.files]


if __name__ == "__main__":
    main()
