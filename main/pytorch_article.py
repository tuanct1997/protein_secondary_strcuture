import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from time import time
from torch import nn,optim
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import tensorflow as tf
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
# Load dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_TRAIN = pd.read_csv('protein-secondary-structure.train', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
DATA_TEST = pd.read_csv('protein-secondary-structure.test', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# i1,l1 = next(iter(trainloader))
# print(i1.shape)
# print(l1.shape)
# print(l1)
# print(a)

amino_map = DATA_TRAIN.copy()
second_map = DATA_TRAIN.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
DATA_TRAIN.fillna(0,inplace = True)
DATA_TEST.fillna(0,inplace = True)

class LSTM(object):
    """docstring for LSTM"""
    def __init__(self):
        super(LSTM, self).__init__()
        self.arg = arg
        

class RNN(object):
    """docstring for RNN"""
    def __init__(self, arg):
        super(RNN, self).__init__()
        self.arg = arg
        
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = torch.nn.Linear(20,1000)
        self.dense2 = torch.nn.Linear(1000, 500)
        self.dense3 = torch.nn.Linear(500,300)
        self.dropout = nn.Dropout(0.5)
        self.dense4 = torch.nn.Linear(300*5, 3)
        self.act = torch.nn.ReLU()
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.act(self.dense3(x))
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.dense4(x)
        # x = self.softmax(x)
        return x



def article_padding(ls):
    ls.insert(0,0)
    ls.insert(0,0)
    ls.insert(len(ls),0)
    ls.insert(len(ls),0)
    return ls


def final_df(df):
    x = []
    y = []
    for idx,rows in df.iterrows():
        chunks = []
        y.extend(rows['label'])
        padd_amino = article_padding(rows['amino'])
        chunks = [padd_amino[x:x+5] for x in range(0, len(padd_amino)-4)]
        x.extend(chunks)

    new_df = pd.DataFrame({'amino':x,'label':y})

    return new_df 
        

#reformat the dataset 
def re_formatdata(data):
    temp = []
    total = []
    labeltemp = []
    label = []
    for idx,rows in data.iterrows():
        if rows['label'] != 0:
            temp.append(rows['amino'])
            labeltemp.append(rows['label'])
        else:
            total.append(temp)
            label.append(labeltemp)
            temp = []
            labeltemp =[]
            continue

    return total,label

def map_int(ls):
    amino_dict = {}
    code = 0
    for i in ls:
        amino_dict[i] = code
        code += 1

    return amino_dict


def encoding_to_int(df,first_map,second_map):
    for idx,rows in df.iterrows():
        row_encode = []
        for code in rows['amino']:
            if code != 0:
                row_encode.append(first_map.get(code))
            else :
                row_encode.append(-10)
        df.at[idx,"amino"] = row_encode
        row_encode = []
        for code in rows['label']:
            if code != 0:
                row_encode.append(second_map.get(code))
            else :
                row_encode.append(-10)
        df.at[idx,"label"] = row_encode
        row_encode = []

    return df

def check_acc(result,prediction):
    count = 0
    for index,val in enumerate(prediction):
        if val == result[index]:
            count += 1

    return count/len(result)



totaltrain,labeltrain= re_formatdata(DATA_TRAIN)
totaltest,labeltest= re_formatdata(DATA_TEST)

amino_map.drop_duplicates(inplace = True)
second_map.drop_duplicates(inplace = True)
amino_map = amino_map.tolist()
second_map = second_map.tolist()

amino_map = map_int(amino_map)
second_map = map_int(second_map)

train = pd.DataFrame({'amino':totaltrain,'label':labeltrain})
test = pd.DataFrame({'amino':totaltest,'label':labeltest})

# padding sequences

processed_train = final_df(train)
processed_test = final_df(test)

processed_train = encoding_to_int(processed_train,amino_map,second_map)
processed_test = encoding_to_int(processed_test,amino_map,second_map)
# .flatten()
amino = np.array(processed_train['amino'].to_list())
label = np.array(processed_train['label'].to_list()).flatten()

x_train, x_test, y_train, y_test = train_test_split(amino, label, test_size=0.2, shuffle = False)

x_train = torch.from_numpy(tf.one_hot(x_train, depth =20).numpy()).to(device)
x_test = torch.from_numpy(tf.one_hot(x_test, depth =20).numpy()).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_test = torch.from_numpy(y_test).to(device)

trainset = TensorDataset(x_train,y_train)
# valset = TensorDataset(Tensor(x_test),Tensor(y_test))


loader = DataLoader(trainset, batch_size = 64)
i1,l1 = next(iter(loader))

amino = tf.one_hot(amino,depth = 20).numpy()
label = tf.one_hot(label, depth = 3).numpy()
# Begin model

# Device for achille - Requirement pytorch 1.7.1, torchivision 0.8.2 and audio 0.7.2
# Achille Cuda 10.1
print(device)
print('!!!!!!!!!')
model = MLP()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_losses = []
val_losses = []

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs,labels = data
        # inputs,labels = data[0].to(device), data[1].to(device)
        # labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs = outputs.permute(0, 2, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i == len(loader)-1 :
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("DONE")
outputs = model(x_test)
print(outputs)
print(outputs.shape)
_, predicted = torch.max(outputs,1)

acc = check_acc(y_test, predicted)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(acc)
print(a)
percentage_ls = []
for idx,val1 in enumerate(outputs):
    count = 0
    for idx1,val2 in enumerate(val1):
        val = torch.argmax(val2)
        print("THIS VAL {}".format(val))
        print("THIS Y {}".format(y_test[idx][idx1]))
        if val == y_test[idx][idx1]:
            count += 1
    leng = [i for i in y_test[idx] if i != 20]
    percentage_ls.append(count/len(leng))

print("FINAL ACC = {}".format(sum(percentage_ls)/len(percentage_ls)))

