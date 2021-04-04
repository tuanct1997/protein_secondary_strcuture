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

# Load dataset
EPOCHS = 100
DATA = pd.read_csv('protein-secondary-structure.train', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
amino_map = DATA.copy()
second_map = DATA.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
DATA.fillna(0,inplace = True)

# fill in 0 paddings
def add_pading(ls,lenght):
    if lenght != len(ls):
        no_padding = lenght - len(ls)
        ls.extend([0]*no_padding)
        return ls
    else:
        return ls


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
            if temp:
                total.append(temp)
                label.append(labeltemp)
            temp = []
            labeltemp =[]
            continue

    return total,label,data

# Create a map char-int
def map_int(ls):
    amino_dict = {}
    code = 1
    for i in ls:
        amino_dict[i] = code
        code += 1

    return amino_dict

# Convert data from char to int
def encoding_to_int(df,first_map,second_map):
    for idx,rows in df.iterrows():
        row_encode = []
        for code in rows['amino']:
            if code != 0:
                row_encode.append(first_map.get(code))
            else :
                row_encode.append(0)

        df.at[idx,"amino"] = row_encode
        row_encode = []
        for code in rows['label']:
            if code != 0:
                row_encode.append(second_map.get(code))
            else :
                row_encode.append(0)
        df.at[idx,"label"] = row_encode
        row_encode = []

    return df

# function to validate percentage of model without 0 paddings
def metrics_protein(result,prediction):
    
    count = 0
    percentage_ls = []
    for idx,val in enumerate(result):
        # prediction[idx] = prediction[:len(val)]
        for idx2,val2 in enumerate(val):
            if val2 == prediction[idx][idx2]:
                count += 1
        percentage = count/len(val)
        count = 0
        print(percentage)
        print('==============')
        percentage_ls.append(percentage)

    return sum(percentage_ls)/len(percentage_ls)

# Model Pytorch
class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
 
            self.fc1 = torch.nn.Linear(128,8)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(8,4)
            self.softmax = torch.nn.Softmax(dim=1)
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.softmax(output)
            return output

# class Net(nn.Module):
# 	"""docstring for Net"""
# 	def __init__(self):
# 		super(Net, self).__init__()
		

# 	def forward(self,x):
# 		x = self.pool(F.relu(self.conv1(x)))
# 		x = self.pool(F.relu(self.conv2(x)))
# 		x = F.relu(self.conv2(x))
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = F.dropout(x, training = self.training)
# 		x = self.fc3(x)
# 		return x

# Train function 
def train(epoch,x_train,y_train,x_test,y_test, opt, loss):


    model.train()
    tr_loss = 0
    x_train, y_train = torch.autograd.Variable(x_train), Variable(y_train)
    x_test, y_test = Variable(x_test), Variable(y_test)

    # clearing the Gradients of the model parameters
    opt.zero_grad()
    
    # prediction for training and validation set

    output_train = model(x_train)
    output_val = model(x_test)

    # computing the training and validation loss
    loss_train = loss(output_train, y_train)
    loss_val = loss(output_val, y_test)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    #if epoch%2 == 0:
        # printing the validation loss
    print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)


# Begin pre-processing data
total,label,data = re_formatdata(DATA)
amino_map.drop_duplicates(inplace = True)
second_map.drop_duplicates(inplace = True)
amino_map = amino_map.tolist()
second_map = second_map.tolist()

amino_map = map_int(amino_map)
second_map = map_int(second_map)


data = pd.DataFrame({'amino':total,'label':label})

data['amino_count'] = data['amino'].apply(lambda x: len(x))

lenght = data['amino_count'].max()

# padding sequences	
for idx,rows in data.iterrows():
    data.at[idx,'amino'] = add_pading(rows['amino'],lenght)
    data.at[idx,'label'] = add_pading(rows['label'],lenght)

data = encoding_to_int(data,amino_map,second_map)
data.drop(['amino_count'],axis = 1, inplace = True)
amino = np.array(data['amino'].tolist())
label = np.array(data['label'].tolist())


x_train, x_test, y_train, y_test = train_test_split(amino, label, test_size=0.2)


# convert to torch format
x_train = F.one_hot(torch.from_numpy(x_train))
x_test = F.one_hot(torch.from_numpy(x_test))
y_train = F.one_hot(torch.from_numpy(y_train))
y_test = F.one_hot(torch.from_numpy(y_test))

print(x_train.shape)
print(y_train.shape)
print(y_test.shape)
print(x_test.shape)
print('==========')

# Begin model
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
	train(epoch,x_train, y_train, x_test, y_test, optimizer, criterion)



