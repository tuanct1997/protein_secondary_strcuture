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
import time

#Check Device for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load dataset
DATA_TRAIN = pd.read_csv('protein-secondary-structure.train', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
DATA_TEST = pd.read_csv('protein-secondary-structure.test', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])


amino_map = DATA_TRAIN.copy()
second_map = DATA_TRAIN.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
DATA_TRAIN.fillna(0,inplace = True)
DATA_TEST.fillna(0,inplace = True)



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(5,64,kernel_size = 1)
        self.dropout = nn.Dropout(0.5)
        self.pool_1 = nn.MaxPool1d(2)
        self.dense = nn.Linear(640,100)
        self.dense2 = nn.Linear(100,50)
        self.dense3 = nn.Linear(50,3)

    def forward(self,x):
        x = self.conv1(x)
        print("!!!!")
        x = torch.relu(x)
        print("!!!!")
        x = self.pool_1(x)
        print("!!!!")
        print(x.shape)
        x = x.view(x.size(0),-1) #flattern
        x = self.dense(x)
        x = self.dense2(x)
        x = self.dropout(x)
        output = self.dense3(x)
        print("!!!!")
        return output

#LSTM MODEL
class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(20, 128) # 20 input size - 128 hidden size
        self.lstm1 = nn.LSTM(128, 64) # 128 input size - 64 hidden size
        self.dropout = nn.Dropout(0.5) # dropout
        self.dense = nn.Linear(64, 3) # Dense - output is 3 classes
        self.act = nn.ReLU()

    def forward(self, x):
        lstm_out, lstm_hidden = self.lstm(x)
        lstm_out, lstm_hidden = self.lstm1(lstm_out)
        lstm_out = lstm_out[:,-1,:] # Take only last timestemps to Linear layers - Many-to-one problem
        lstm_out = self.act(lstm_out)
        drop_out = self.dropout(lstm_out)
        output = self.dense(drop_out)
        return output


class RNN(nn.Module):
    """docstring for RNN"""
    def __init__(self):
        super(RNN, self).__init__()
        # ,nonlinearity = 'relu' -- Not good as default tanh - this to declare the activation function of RNN
        self.rnn = nn.RNN(20,500) # 20 input size - 128 hidden size
        self.rnn1 = nn.RNN(500,300)# 128 input size - 64 hidden size
        self.dense = nn.Linear(300,3)# Dense - output is 3 classes
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)# dropout

    def forward(self,x):
        rnn_out, rnn_hidden = self.rnn(x)
        rnn_out, rnn_hidden = self.rnn1(rnn_out)
        rnn_out = rnn_out[:,-1,:]# Take only last timestemps to Linear layers - Many-to-one problem
        rnn_out = self.act(rnn_out)
        rnn_out = self.dropout(rnn_out)
        output = self.dense(rnn_out)

        return output
        
        
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = nn.Linear(20,500)# 20 input size - 500 hidden size
        self.dense2 = nn.Linear(500, 300)# 500 input size - 300 ouput size
        self.dense3 = nn.Linear(300,100)# 300 input size - 100 ouput size
        self.dropout = nn.Dropout(0.5)
        self.dense4 = nn.Linear(100*5, 3)# 100Nodes * flatten_size(timestemps) input size - 3 classes output
        self.act = nn.ReLU()
        # no need of softmax when we use crossentropyloss / if use BCELoss , active the softmax
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.act(self.dense3(x))
        x = self.dropout(x)
        x = x.view(x.size(0),-1) #flattern
        x = self.dense4(x)
        # no need of softmax when we use crossentropyloss / if use BCELoss , active the softmax
        # x = self.softmax(x)
        return x



# zero_padding window_slide
def article_padding(ls):
    # Adđ 2 zero at the begin and 2 at the end 
    ls.insert(0,0)
    ls.insert(0,0)
    ls.insert(len(ls),0)
    ls.insert(len(ls),0)
    return ls


# Final dataset after apply window_slide
def final_df(df):
    #input df : dataframe
    x = []
    y = []
    # loop through each rows of dataframe
    for idx,rows in df.iterrows():
        chunks = []
        y.extend(rows['label'])
        padd_amino = article_padding(rows['amino'])
        # create a list of all sub-list 
        # scan from begin 0 to the last in order to take all 5 lenght sequence
        chunks = [padd_amino[x:x+5] for x in range(0, len(padd_amino)-4)]
        x.extend(chunks)

    #create final df
    new_df = pd.DataFrame({'amino':x,'label':y})

    return new_df 
        

#reformat the dataset 
def re_formatdata(data):
    temp = []
    total = []
    labeltemp = []
    label = []
    # loop through the raw dataset
    for idx,rows in data.iterrows():
        # if not the end of sequence still apped into row - else : reach the end , switch store next element of ls
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

# create the char:int map
def map_int(ls):
    final_map = {}
    code = 0
    for i in ls:
        final_map[i] = code
        code += 1

    return final_map


# encoding base on the map
def encoding_to_int(df,first_map,second_map):
    for idx,rows in df.iterrows():
        row_encode = []
        for code in rows['amino']:
            # if that is zero-padding - encode to -10 to make it far from other when model calculate
            # else -> check the map and encode
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

# check accuracy by number of right predicted
def check_acc(result,prediction):
    count = 0
    # check each prediction
    for index,val in enumerate(prediction):
        # if predict correct => plus 1 point
        if val == result[index]:
            count += 1
    # return the percentage accuracy
    return count/len(result)


# First make a list of each sequence from .train and .test
totaltrain,labeltrain= re_formatdata(DATA_TRAIN)
totaltest,labeltest= re_formatdata(DATA_TEST)

# create unique char of amino acid
amino_map.drop_duplicates(inplace = True)

# create unique char of secondary
second_map.drop_duplicates(inplace = True)
amino_map = amino_map.tolist()
second_map = second_map.tolist()

amino_map = map_int(amino_map)
second_map = map_int(second_map)

# make dataframe of full sequence 
#exp :
# amino - label 
# ABCDE    _,_,h,h,_
train = pd.DataFrame({'amino':totaltrain,'label':labeltrain})
test = pd.DataFrame({'amino':totaltest,'label':labeltest})

# padding sequences
processed_train = final_df(train)
processed_test = final_df(test)

# final dataframe of 5 lenght sequence 
#exp: 
# amino - label 
# 00ABC -  _
# 0ABCD -  _
# ABCDE -  h
processed_train = encoding_to_int(processed_train,amino_map,second_map)
processed_test = encoding_to_int(processed_test,amino_map,second_map)


amino = np.array(processed_train['amino'].to_list())
#need to flatten before one-hot encoding
label = np.array(processed_train['label'].to_list()).flatten()

# keras split
x_train, x_test, y_train, y_test = train_test_split(amino, label, test_size=0.2, shuffle = False)

# any dataset should be push to deivce(if we have GPU => automatically push)
x_train = torch.from_numpy(tf.one_hot(x_train, depth =20).numpy()).to(device)
x_test = torch.from_numpy(tf.one_hot(x_test, depth =20).numpy()).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_test = torch.from_numpy(y_test).to(device)

# Create TensorDataSet
trainset = TensorDataset(x_train,y_train)
# valset = TensorDataset(Tensor(x_test),Tensor(y_test))

# Create batch by DataLoader
loader = DataLoader(trainset, batch_size = 64)
#check if correctly
i1,l1 = next(iter(loader))
print(x_train[0])
print(y_train.shape)
# Begin model

# Device for achille - Requirement pytorch 1.7.1, torchivision 0.8.2 and audio 0.7.2
# Achille Cuda 10.1
# Please check PyTorch for suitable version for your GPU
#check device
print(device)
print('!!!!!!!!!')
#Create model for 3 different architecture
# Need to push to device in case we have GPU
model = MLP()
model2 = LSTM()
model3 = RNN()
model4 = CNN()
model3.to(device)
model2.to(device)
model.to(device)
model4.to(device)

# Declare Cross Entropy Loss with Adam optimizer with learning rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_losses = []
val_losses = []
paitent = 0
old = 0

# Start time count for only training in order to be fair when compare with keras
start_time = time.time()

#100 epochs
for epoch in range(100):
    running_loss = 0.0
    # begin update with each mini-batch
    for i, data in enumerate(loader, 0):
        inputs,labels = data
        print(inputs.shape)
        optimizer.zero_grad()
        # remember to replace model if we want another network 
        #model => MLP
        # model2 => LSTM
        # model3 => RNN
        outputs = model4(inputs)
        # print(outputs.shape)
        # print(labels)
        # outputs = outputs.permute(0, 2, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #reach to the end of mini-batch
        if i == len(loader)-1 :
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 270))

    # !!!!!!!!!!!!!!!!!!!!!!BELOW SHOULD BE EARLY STOPPING - USE WHEN NEEDED......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         print(running_loss/270 - old)
    #         if running_loss/270 - old > 1e-4 :
    #             print('aaaaa')
    #             paitent += 1
    #         else :
    #             print('23232')
    #             paitent = 0
    #         if running_loss/270 < old :
    #             old = running_loss / 270
    #         running_loss = 0.0
    # if paitent == 5:
    #     print("EARLY STOP")
    #     break
print("--- %s seconds ---" % (time.time() - start_time))
print("DONE")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test - reaplace the model as the model we use for train above
# x-test is val set
outputs = model4(x_test)
print(outputs)
print(outputs.shape)
#convert output to final class
_, predicted = torch.max(outputs,1)

# Test set
aminotest = np.array(processed_test['amino'].to_list())
labeltest = torch.from_numpy(np.array(processed_test['label'].to_list()).flatten()).to(device)
aminotest = torch.from_numpy(tf.one_hot(aminotest, depth =20).numpy()).to(device)
outputs_test = model3(aminotest)
_, predictedtest = torch.max(outputs_test,1)

val_acc = check_acc(y_test, predicted)
test_acc = check_acc(labeltest, predictedtest)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("VAL_ACC : {}".format(val_acc))
print("TEST_ACC : {}".format(test_acc))

