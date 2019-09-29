import cv2
import opencv_util
from opencv_util import color


def handle_key(key,img):
    if key == opencv_util.KEY_LEFT :
        print('KEY_LEFT')
    elif key == opencv_util.KEY_UP :
        print('KEY_UP')
    elif key == opencv_util.KEY_RIGHT :
        print('KEY_RIGHT')
    elif key == opencv_util.KEY_DOWN :
        print('KEY_DOWN')

def main():
    cv2.startWindowThread()

    red = cv2.imread('img/dog.jpg')
    red = opencv_util.resize_square_keep_aspect(red, 512, color('#FFFF00'))
    opencv_util.closeable_imshow('test', red, handle_key)


if __name__ == "__main__":
    main()
