import numpy as np
from numpycsv import numpycsv

csv = numpycsv('sample.csv')
# numpy like indexies
print(csv[:, 0])
# iteratable
for row in csv:
    print(row)

# convert ndarray
array = np.array(csv)
print("convert ndarray")
print(array)

# numeric csv
csv = numpycsv('sample.csv', forceint=True)
print(csv)

# 最大値の座標
# https://qiita.com/tktktks10/items/f85aeef3321f6cbbd368
print(np.unravel_index(np.argmax(csv), csv.shape))

# csvの型がndarrayじゃないと失敗するので
# 一度変換する必要がある
# 変更すると影響を受けるarrayと変更を受けないcopyarrayを用意
csv2 = csv.copyarray()
csv2[csv2 > 1] *= 10
print(csv2)
