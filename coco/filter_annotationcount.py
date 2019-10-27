import coco
from coco import Annotation, AnnotationId
import argparse
import time


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # http://ja.pymotw.com/2/argparse/
    parser.add_argument("inputfile", action="store")
    parser.add_argument('-s', dest='start', type=int, default=0)
    parser.add_argument('-c', dest='count', type=int, default=10)
    parser.add_argument('-o', dest='outputfile',
                        type=str, default='output.json')

    return parser.parse_args()


def main():
    args = getargs()
    begin_time = int(round(time.time() * 1000))
    with coco.coco(args.inputfile) as data:
        end_time = int(round(time.time() * 1000))
        print(f'finish load {end_time-begin_time} ms', flush=True)
        all_ids = data.np_annotations[:, Annotation.ID]
        filterd_ids = all_ids[args.start:args.start + args.count]
        print('begin filter')
        begin_time = int(round(time.time() * 1000))
        c2 = data & AnnotationId(filterd_ids)
        end_time = int(round(time.time() * 1000))
        print(f'finish filter {end_time-begin_time} ms', flush=True)
        c2.save(args.outputfile)


if __name__ == "__main__":
    main()
