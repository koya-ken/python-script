import coco
from coco import Annotation
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
    filename = os.path.basename(args.inputfile)
    with coco.coco(args.inputfile) as data:
        np_annotations = data.np_annotations
        num_keypoints = np_annotations[:, Annotation.NUM_KEYPOINTS]
        num_keypoints_counter = collections.Counter(num_keypoints)
        analyze_num_keypoints = np.array(num_keypoints_counter.most_common())
        # https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure.add_subplot
        # window title
        # fig = plt.figure(num=filename)

        fig = plt.figure(figsize=(12, 6))
        plots = fig.subplots(1, 2)
        x = analyze_num_keypoints[:, 0]
        y = analyze_num_keypoints[:, 1]
        total_annotations = np.sum(y)
        y = y / total_annotations * 100
        for tmpx, tmpy in zip(x, y):
            plots[0].text(tmpx, tmpy, f"{tmpy:.1f}", ha='center', va='bottom')
        plots[0].set_title(f"{filename}\ntotal annotation={total_annotations}")
        plots[0].bar(x, y, tick_label=x)
        plots[0].set_xlabel('keypoints')
        plots[0].set_ylabel('keypointnum (percent)')
        # descending sort
        # sorted_indexies = np.argsort(-y)
        # https://pythondatascience.plavox.info/matplotlib/%E5%86%86%E3%82%B0%E3%83%A9%E3%83%95
        plots[1].pie(y, labels=x, counterclock=False,
                     startangle=90, autopct="%1.1f%%")
        plt.show()


if __name__ == "__main__":
    main()
