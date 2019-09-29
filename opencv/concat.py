import cv2

red = cv2.imread('img/red.png')
green = cv2.imread('img/green.png')
blue = cv2.imread('img/blue.png')
yellow = cv2.imread('img/yellow.png')

# 結合したいものをリストで渡さないといけないのに注意する
rg = cv2.hconcat([red, green])
by = cv2.hconcat([blue, yellow])

dstimg = cv2.vconcat([rg, by])
cv2.imwrite('img/square.png', dstimg)

# https://note.nkmk.me/python-opencv-hconcat-vconcat-np-tile/
