import cv2
import argparse
import numpy as np


def getargs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # http://ja.pymotw.com/2/argparse/
    parser.add_argument('-i', '--inputfile', required=True)
    parser.add_argument('-o', dest='outputfile', type=str, default=None)
    parser.add_argument('-s', dest='filtersize', type=int, default=None)

    return parser.parse_args()


def main():
    args = getargs()
    # 元の画像を読み込む
    org_img = cv2.imread(args.inputfile, cv2.IMREAD_GRAYSCALE)
    print(org_img.dtype)
    print(org_img.shape)
    if args.filtersize:
        # kernel = np.ones((args.filtersize, args.filtersize), np.float32)/25
        # org_img = cv2.filter2D(org_img, -1, kernel)
        org_img = cv2.GaussianBlur(org_img, ksize=(args.filtersize, args.filtersize), sigmaX=10)
    # エッジ抽出
    # 第2,3引数はヒステリシスを使ったしきい値処理に使う minVal と maxVal をそれぞれ指定します．
    # 第4引数は画像の勾配を計算するためのSobelフィルタのサイズ aperture_size で，デフォルト値は3です．
    # 最後の引数は勾配強度を計算するための式を指定する L2gradient です．
    # True を指定すると上述した式を使い，より精度が高いエッジ強度を計算します．
    # そうでなければ以下の式を使ってエッジ強度を計算します:
    # Edge\_Gradient \; (G) = |G_x| + |G_y|.デフォルトのフラグは False を指定しています．
    canny_img = cv2.Canny(org_img, 50, 110)
    out_img = cv2.hconcat([org_img, canny_img])

    cv2.imshow('edge', out_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
