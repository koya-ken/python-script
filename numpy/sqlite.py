import sqlite3
import numpy as np
import io
# import zlib as compress
import bz2 as compress

# https://kotaeta.com/56789628


def empty():
    pass


# empty.compress = lambda x: x
# empty.decompress = lambda x: x
# compress = empty


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(compress.compress(out.read()))


def convert_array(text):
    out = io.BytesIO(compress.decompress(text))
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

x = np.arange(12).reshape(2, 6)

con = sqlite3.connect("test.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()

cur.execute("create table if not exists test (arr array)")

size = 10000
for i in range(size):
    cur.execute("insert into test (arr) values (?)", (x, ))

# for row in cur.execute("select arr from test"):
#     print(row[0])
cur.execute("select arr from test")
data = cur.fetchone()[0]
# data = cur.fetchall()

print(data)
con.commit()
