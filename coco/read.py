import msgpack


def t():
    with open('data.msgpack', 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    # print(data['images'])
