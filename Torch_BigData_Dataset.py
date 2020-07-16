import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
import linecache
import time,timeit
import itertools
import memory_profiler

class TorchClass(Dataset):
    def __init__(self,num_index,out_dic,chinksize=100,name='invoice_train.csv',
                 label='precipProbability',categorical_list=None):
        self.reader = pd.read_csv(name,header=0,iterator=True)
        self.chunksize = chinksize
        self.label= label
        self.categorical = categorical_list
        self.num_index = num_index
        self.out_dic=out_dic
        self.start= 0
    def __len__(self):
        """
        :return: it should return size of dataset
         since I know it already which is 500000 rows
         and chunksize I considered is 100 so the length is 500000/100
        """
        return int(4400000 / self.chunksize)
    def __getitem__(self, item):
        tmp= self.reader.get_chunk(self.chunksize)
        tmp.dropna(axis=0,inplace=True)
        # y = tmp.pop(self.label)
        # x = tmp.select_dtypes(exclude='object')
        x= tmp.iloc[:,self.num_index].select_dtypes(include=['int64','float64'])
        y= tmp.counter_type.apply(lambda x: self.out_dic[x]).values
        return x,y

class TorchDataSetLineCache(torch.utils.data.Dataset):
    def __init__(self,num_index,out_dict,file_name='invoice_train.csv'):
        self.file_name= file_name
        self.num_index = num_index
        self.out_dict = out_dict
    def __getitem__(self, item):
        line = linecache.getline(self.file_name,item+1)
        reader= csv.reader([line])
        tmp=next(reader)
        x= list(map(lambda x: float(x) if x.replace('.','').isnumeric() else 0.0,[tmp[i] for i in self.num_index]))
        y=0
        if tmp[-1] in ['ELEC','GAZ']:
            y=self.out_dict[tmp[-1]]
        return x,y
    def __len__(self):
        return 4400000

cat_index = [0, 1, 15]
num_index = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
mydic={'ELEC': 0 , 'GAZ':1}

dataset1= TorchClass(num_index=num_index,out_dic=mydic,chinksize=1000)
loader=  DataLoader(dataset1,batch_size=1,shuffle=True)

dataset2= TorchDataSetLineCache(num_index=num_index,out_dict=mydic)
loader2= DataLoader(dataset2,batch_size=1000,shuffle=True)


class NonLinearRegression(torch.nn.Module):
    def __init__(self,n_input,n_hid,n_out):
        super(NonLinearRegression,self).__init__()
        self.n_input = n_input
        self.n_out = n_out
        self.n_hid = n_hid
        self.linear= torch.nn.Linear(in_features=self.n_input,out_features=self.n_hid)
        self.out = torch.nn.Linear(in_features=self.n_hid,out_features=self.n_out)
    def forward(self,x):
        x= self.linear(x)
        x= torch.relu(x)
        x= torch.sigmoid(self.out(x))
        return x

mymodel=NonLinearRegression(13,10,1)
optim= torch.optim.Adam(mymodel.parameters(),lr=.01)
loss= torch.nn.BCELoss()
def train_step(x,y):
    out= mymodel(x)
    optim.zero_grad()
    l= loss(y,out)
    l.backward()
    optim.step()
    return l.item()



# @profile
def func1():
    print('first function using pandas')
    mymodel = NonLinearRegression(13, 10, 1)
    optim = torch.optim.Adam(mymodel.parameters(), lr=.01)
    loss = torch.nn.BCELoss()
    def train_step(x, y):
        out = mymodel(x.squeeze(0).float())
        optim.zero_grad()
        l = loss(out,y.view(-1,1).float())
        l.backward()
        optim.step()
        return l.item()

    loss_ = []
    start1 = time.time()
    for x,y in loader:
        loss_.append(train_step(x,y))
    duration1 = time.time() - start1
    print('duration for chunk method is {}'.format(duration1))
    return loss_, duration1

# @profile
def func2():
    mymodel = NonLinearRegression(13, 10, 1)
    optim = torch.optim.Adam(mymodel.parameters(), lr=.01)
    loss = torch.nn.BCELoss()
    def train_step(x, y):
        out = mymodel(torch.stack(x,-1).float())
        optim.zero_grad()
        l = loss(out,y.float())
        print(l.item())
        l.backward()
        optim.step()
        return l.item()

    loss_ = []
    start2 = time.time()
    for x, y in loader2:
        loss_.append(train_step(x,y))
    duration2 = time.time() - start2
    print('duration for cache method is {}'.format(duration2))
    return loss_, duration2

# func1()

if __name__=='__main__':
    func1()

a=1