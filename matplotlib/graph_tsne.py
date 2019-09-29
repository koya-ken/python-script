import os
import numpy as np
import cv2
from sklearn.manifold import TSNE
from sklearn import preprocessing

import plotly.offline as offline
import plotly.graph_objs as go

# https://www.mathgram.xyz/entry/plotly/tsne/example

# 画像の前処理．標準化やらL2正規化やら．
def preprocess_image(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (size, size), cv2.INTER_LINEAR).astype("float")
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    timg = normalized.reshape(np.prod(normalized.shape))
    result = timg/np.linalg.norm(timg) 
    result[np.isnan(result)] = 0
    return result

ROOT = "./img"
ls = os.listdir(ROOT)

# 名前からラベルを持って来ます．
obj_ls = [name.split("_")[0] for name in ls]

ALL_IMAGE_PATH = [ROOT+"/"+path for path in ls]

# 全画像に対して前処理する
preprocess_images_as_vecs = [preprocess_image(path, 512) for path in ALL_IMAGE_PATH]

# tsneを実行
tsne = TSNE(
    n_components=2, #ここが削減後の次元数です．
    init='random',
    random_state=101,
    method='barnes_hut',
    n_iter=1000,
    verbose=2
).fit_transform(preprocess_images_as_vecs)


# # 3Dの散布図が作れるScatter3dを使います．
# trace1 = go.Scatter3d(
#     x=tsne[:,0], # それぞれの次元をx, y, zにセットするだけです．
#     y=tsne[:,1],
#     z=tsne[:,2],
#     mode='markers',
#     marker=dict(
#         sizemode='diameter',
#         colorscale = 'Portland',
#         line=dict(color='rgb(255, 255, 255)'),
#         opacity=0.9,
#         size=2 # ごちゃごちゃしないように小さめに設定するのがオススメです．
#     )
# )

# data=[trace1]
# layout=dict(height=700, width=600, title='coil-20 tsne exmaple')
# fig=dict(data=data, layout=layout)
# offline.plot(fig, filename='tsne_example')

# tsneには2dまで落とし込んだarrayが入っている想定です．

trace = go.Scatter(
    x=tsne[:,0],
    y=tsne[:,1],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = preprocessing.LabelEncoder().fit_transform(obj_ls),
        colorscale = 'Portland',
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9,
        size=4
    )
)

data=[trace]
layout=dict(height=800, width=800, title='coil-20 tsne exmaple 2D')
fig=dict(data=data, layout=layout)
offline.plot(fig, filename='tsne2D_example')