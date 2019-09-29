import cv2
import numpy as np

# http://pynote.hatenablog.com/entry/opencv-resize
# アスペクト比を固定して、幅が指定した値になるようリサイズする。

ORD_ESCAPE = 0x1b
KEY_LEFT = 0x250000
KEY_UP = 0x260000
KEY_RIGHT = 0x270000
KEY_DOWN = 0x280000

def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)

# アスペクト比を固定して、高さが指定した値になるようリサイズする。


def scale_to_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def resize_square_keep_aspect(img, size, border_rgb_color=(0, 0, 0)):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(border_rgb_color))
    # Fill image with color
    f[:] = color

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
    return scale_to_height(f, size)

# https://gist.github.com/matthewkremer/3295567


def color(hex):
    hex = hex.lstrip('#')
    return (int(hex[:2], 16), int(hex[2:4], 16), int(hex[4:], 16))

# https://stackoverflow.com/questions/4337902/how-to-fill-opencv-image-with-one-solid-color


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

# https://gist.github.com/seraphy/f36e04aa4379e465ff41266319b5238a
# https://stackoverflow.com/questions/35003476/opencv-python-how-to-detect-if-a-window-is-closed


def _is_visible(winname):
    try:
        return cv2.getWindowProperty(winname, 0) >= 0
    except cv2.error:
        return False


def closeable_imshow(winname, img, handle_key_callback=None, break_key=ORD_ESCAPE):
    while True:
        cv2.imshow(winname, img)
        key = cv2.waitKeyEx(10)

        if key == break_key:
            break
        elif key != -1 and handle_key_callback is not None:
            handle_key_callback(key, img)

        if not _is_visible(winname):
            break

    for i in range(1, 10):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
