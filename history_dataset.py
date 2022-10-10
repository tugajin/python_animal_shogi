from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from single_network import *
import numpy as np
import time
import json
from db import *

class HistoryDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = []
        for path in root:
            with path.open(mode='rb') as f:
                data.extend(pickle.load(f))
        data2 = []
        for d in data:
            dd = np.array(d[0])
            file, rank, channel = DN_INPUT_SHAPE
            dd = dd.reshape(channel, file, rank)
            data2.append([dd, d[1]])
        self.data = data2
    # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        data = self.data[index][0]
        y_deep = self.data[index][1]
        return data, y_deep

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.data)

class HistoryDataset2(Dataset):
    def __init__(self, num = 1000000):
        super().__init__()
        n = select_count_all()
        print(f"num:{n} select:{num}")
        l = [i+1 for i in range(n)]
        if num < len(l):
            indexs = random.sample(l, num)
        else:
            indexs = l
        data2 = []
        conn, cur = create_conn()
        for index in indexs:
            dd, result = select2(cur, index)
            dd = json.loads(dd)
            dd = np.array(dd)
            file, rank, channel = DN_INPUT_SHAPE
            dd = dd.reshape(channel, file, rank)
            data2.append([dd, result])
        self.data = data2
        close_conn(conn)
    # ここで取り出すデータを指定している
    def __getitem__(self, index) :
        data = self.data[index][0]
        y_deep = self.data[index][1]
        return data, y_deep

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = HistoryDataset2()