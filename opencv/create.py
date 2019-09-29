import cv2
import numpy as np

# pythonのopencvはnumpyでできている
black = np.zeros(100 * 100).reshape(100, 100)
cv2.imwrite('img/black.png', black)

# reshapeは行数、列数 = 高さ、幅
black2 = np.zeros(100 * 50).reshape(100, 50)
cv2.imwrite('img/black2.png', black2)
