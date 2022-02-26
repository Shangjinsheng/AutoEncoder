import sys
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn
import os
from collections import Counter
from torch.utils.data import Dataset,DataLoader



args = sys.argv
file_dir = args[args.index('-input')+1]#由terminal输入决定file的位置
library_dir = args[args.index('-library')+1]#由terminal输入决定library的位置

aa_dict_dir =library_dir+'/Atchley_factors.csv'#atchley编码的存放地址

aa_dict_atchley = dict()#创造了一个关于tcr的氨基酸的embedding字典

with open(aa_dict_dir,'r') as aa:
    aa_reader = csv.reader(aa)
    next(aa_reader,None)#这一段的作用是跳过开头第一行
    for rows in aa_reader:
        aa_name = rows[0]
        aa_factor = rows[1:len(rows)]
        aa_dict_atchley[aa_name] = np.asarray(aa_factor,dtype = 'float')

# print(aa_dict_atchley)
# print(len(aa_dict_atchley))

# test1 = pd.read_csv(file_dir,header = 0)

# test2 = test1.sort_values('CDR3').reset_index(drop = True)
# test3 = test2['CDR3'].tolist()


def preprocess(filedir):
    print("Processing:"+filedir)
    
    if not os.path.exists(filedir):
        print('非法路径' + filedir)
        return 0
    
    dataset = pd.read_csv(filedir,header=0)#header=0的作用将第一行作为列名
    dataset = dataset.sort_values('CDR3').reset_index(drop = True)#按照CDR3的顺序进行排列
    dataset = dataset.dropna()#去掉缺失值

    TCR_list = dataset['CDR3'].tolist()
    print('TCR_list长度为：',len(TCR_list))
    return TCR_list



def aamapping_TCR(peptideSeq,aa_dict):
    peptideArray = []
    if len(peptideSeq)>80:
        print('Length:'+str(len(peptideSeq))+'over bound!')
        peptideSeq = peptideSeq[0:80]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeq:'+peptideSeq)#如果在Atchley编码字典中没有该氨基酸则证明不合适
            peptideArray.append(np.zeros(5,dtype='float32'))
    for i in range(80-len(peptideSeq)):#不够80的地方补成0
        peptideArray.append(np.zeros(5,dtype='float32'))

    return np.asarray(peptideArray)


def TCRMap(dataset,aa_dict):
    pos = 0
    TCR_counter = Counter(dataset)
    print('这玩意儿的长度为',len(TCR_counter))
    TCR_array = np.zeros((len(TCR_counter),1,80,5),dtype = np.float32)
    for sequence,length in TCR_counter.items():
        # TCR_array[pos:pos+length] = np.repeat(aamapping_TCR(sequence,aa_dict).reshape(1,1,80,5),length,axis=0)
        TCR_array[pos] = aamapping_TCR(sequence,aa_dict).reshape(1,1,80,5)
        pos += 1
    print('TCRMap done!!!')
    return TCR_array


TCR_list = preprocess(file_dir)
TCR_array = TCRMap(TCR_list,aa_dict_atchley)
# print(TCR_array)
print('TCR_arrary的shape为:',TCR_array.shape)
print(type(TCR_array))
print(len(TCR_array))

class Mydataset(Dataset):
    def __init__(self,tcr_array):
        self.tcr_array = tcr_array
    
    def __getitem__(self,index):
        return self.tcr_array[index]
    
    def __len__(self):
        return len(self.tcr_array)
    

train_set = Mydataset(TCR_array)
train_loader = DataLoader(dataset =train_set,batch_size = 16,shuffle = True )
print(len(train_loader))