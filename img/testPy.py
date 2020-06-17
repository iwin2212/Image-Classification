import os
from const import *


# detail about train set
def info(train_path):
    print(train_path)
    dict = {}
    listLabel = os.listdir(train_path)
    for each in listLabel:
        numberFiles = len(os.listdir(os.path.join(train_path, each)))
        dict[each] = numberFiles
    return dict


print(info(train_path))
