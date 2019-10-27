import coco
from coco import Image, ImageId
import argparse
import time


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # http://ja.pymotw.com/2/argparse/
    parser.add_argument("inputfile", action="store")
    parser.add_argument('-f', dest='filter', required=True)
    parser.add_argument('-o', dest='outputfile',
                        type=str, default='filtered.json')

    return parser.parse_args()


def main():
    args = getargs()
    with coco.coco(args.inputfile) as data:
        c2 = data.filter(args.filter)
        c2.save(args.outputfile)


if __name__ == "__main__":
    main()
