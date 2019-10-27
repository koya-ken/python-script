import coco
from coco import Image, ImageId
import argparse
import time


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # http://ja.pymotw.com/2/argparse/
    parser.add_argument("inputfiles", nargs=2)
    parser.add_argument('-o', dest='outputfile',
                        type=str, default='merged.json')

    return parser.parse_args()


def main():
    args = getargs()
    begin_time = int(round(time.time() * 1000))
    with coco.coco(args.inputfiles[0]) as c1,  coco.coco(args.inputfiles[1]) as c2:
        end_time = int(round(time.time() * 1000))
        print(f'finish load {end_time-begin_time} ms', flush=True)
        begin_time = int(round(time.time() * 1000))
        print('begin merge')
        c3 = c1 | c2
        end_time = int(round(time.time() * 1000))
        print(f'finish merge {end_time-begin_time} ms', flush=True)
    c3.save(args.outputfile)


if __name__ == "__main__":
    main()
