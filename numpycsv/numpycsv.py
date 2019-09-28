import csv
import numpy as np
import operator


class numpycsv(object):
    def __init__(self, filename, exists_header=True, forceint=False, default_numeric_value=0):
        self.__filename = filename
        self.__exists_header = exists_header
        self.__forceint = forceint
        self.__default_numeric_value = default_numeric_value
        self.__load()

    def __load(self):
        with open(self.__filename) as f:
            reader = csv.reader(f)
            if self.__exists_header:
                self.__header = np.array(next(reader))
            if self.__forceint:
                self.__rows = [list(
                    map(lambda cell: int(cell) if cell.isnumeric() else self.__default_numeric_value, row)) for row in reader]
            else:
                self.__rows = [row for row in reader]

        print(self.__rows)
        self.__csv_matrix = np.array(self.__rows)
        self.shape = self.__csv_matrix.shape

    def header(self):
        return self.__header[:]
    
    def array(self):
        return self.__csv_matrix

    def copyarray(self):
        return self.__csv_matrix.copy()

# iterator access
    def __iter__(self):
        return iter(self.__csv_matrix)

# index access like numpy
    def __getitem__(self, index):
        return self.__csv_matrix[index]

# tostring numpy array
    def __str__(self):
        return str(self.__csv_matrix)

# http://sinhrks.hatenablog.com/entry/2015/09/21/223439
    # use for np.array(csv)
    def __array__(self):
        return self.__csv_matrix

# operator like numpy
    def __lt__(self, b):
        return self.__csv_matrix < b

    def __le__(self, b):
        return self.__csv_matrix <= b

    def __eq__(self, b):
        return self.__csv_matrix == b

    def __ne__(self, b):
        return self.__csv_matrix != b

    def __ge__(self, b):
        return self.__csv_matrix >= b

    def __gt__(self, b):
        return self.__csv_matrix > b

    def __mul__(self, b):
        return self.__csv_matrix * b

    def __imul__(self, b):
        print('imul')
        self.__csv_matrix *= b
        return self.__csv_matrix

    def __matmul__(self, b):
        return self.__csv_matrix @ b

    def __imatmul__(self, b):
        self.__csv_matrix @= b
        return self.__csv_matrix
