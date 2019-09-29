import argparse


def get_args():
    """コマンドライン引数を取得する
    """
    parser = argparse.ArgumentParser('opencv image viewer')
    parser.add_argument('-i', '--input', dest='input',
                        required=True, help='input file or directory')
    parser.add_argument('-w', '--viewer-width', dest='width',
                        default=512, help='viewer width')
    parser.add_argument('-e', '--extension', action='append',
                        default=['jpg'], help='expect extention')
    parser.add_argument('-n', '--show-image-name',
                        default=True, help='show image name')
    parser.add_argument('--view-image-count',
                        default=1, help='view image count')
    # parser.add_argument('-f', '--format', dest='format', default='csv', choices=('csv', 'tsv'), help='input file or directory')

    return parser.parse_args()


def main():
    args = get_args()
    print(args.input)
    print(args.extension)
    pass


def get_image_count():
    """ 画像の総数を取得する
    """
    pass


def get_image(index):
    """ 指定したindexの画像を取得する
    """
    pass

def image_preprocess(img,index):
    pass

def image_afterprocess(img,index):
    pass

def show_image():
    """ 画像を表示する
    """
    pass


if __name__ == "__main__":
    main()
