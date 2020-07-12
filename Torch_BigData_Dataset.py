import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv

class TorchClass(Dataset):
    def __init__(self,chinksize=100,name='HomeC.csv',
                 label='precipProbability',categorical_list=None):
        self.reader = pd.read_csv(name,header=0,iterator=True)
        self.chunksize = chinksize
        self.label= label
        self.categorical = categorical_list
        self.start= 0
    def __len__(self):
        """
        :return: it should return size of dataset
         since I know it already which is 500000 rows
         and chunksize I considered is 100 so the length is 500000/100
        """
        return 5000
    def __getitem__(self, item):
        tmp= self.reader.get_chunk(self.chunksize)
        # y = tmp.pop(self.label)
        x = tmp.select_dtypes(exclude='object')
        return x.values

dataset= TorchClass()
loader=  DataLoader(dataset,batch_size=1,shuffle=True)