import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import pickle
from sklearn.model_selection import StratifiedKFold,TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.datasets import mnist

# pre-processing dataset

DATA_TRAIN = pd.read_csv('protein-secondary-structure.train', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
DATA_TEST = pd.read_csv('protein-secondary-structure.test', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])

# DATA = pd.concat([DATA_TRAIN,DATA_TEST])
# (trainX, trainy), (testX, testy) = mnist.load_data()
# trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
# print(trainy.shape)
# # print(trainy[1])
# trainy = to_categorical(trainy)
# print(trainy.shape)
# print(trainy)
# print(a)

amino_map = DATA_TRAIN.copy()
second_map = DATA_TRAIN.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
DATA_TRAIN.fillna(0,inplace = True)
DATA_TEST.fillna(0,inplace = True)


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

def metrics_protein(result,prediction):
    
    count = 0
    percentage_ls = []
    for idx,val in enumerate(result):
        # prediction[idx] = prediction[:len(val)]
        for idx2,val2 in enumerate(val):
            if val2 == prediction[idx][idx2] and val2 != 20:
                count += 1
        leng = [i for i in val if i != 20]
        percentage = count/len(leng)
        count = 0
        percentage_ls.append(percentage)

    return sum(percentage_ls)/len(percentage_ls)

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
# print(x_train[0])
# print(a)
# x_train = to_categorical(x_train)
# x_test = to_categorical(x_test)
# y_test = to_categorical(y_test)
x_train = tf.one_hot(x_train, depth =20).numpy()
x_test = tf.one_hot(x_test, depth =20).numpy()
# y_train = to_categorical(y_train)
y_train = tf.one_hot(y_train, depth =3).numpy()
y_test = tf.one_hot(y_test, depth =3).numpy()

# y_train = y_train.reshape((y_train.shape[0],1,1,3))
# y_test = y_test.reshape((y_test.shape[0],1,1,3))
print(y_train.shape)
print(y_train)
# print(y_train[0])

cv_score = []


model = keras.Sequential()
# model.add(layers.Dense(128, input_shape = (498,21) , activation ='relu'))
# model.add(layers.Dense(8,activation ='relu'))
# model.add(layers.Masking(mask_value=0., input_shape=20))
# model.add(layers.LSTM(128,return_sequences = False, input_shape = (5,20)))#recurrent layer , 128 neurons
model.add(layers.Bidirectional(layers.GRU(64,return_sequences=True, activation = 'relu'), input_shape = (5,20)))#recurrent layer 1, 64 neurons
model.add(layers.Bidirectional(layers.GRU(32, return_sequences=True, activation = 'relu'))) #recurrent layer 2, 32 neurons
# model.add(layers.Bidirectional(layers.GRU(8,return_sequences=True))) #recurrent layer 3, 16 neurons
model.add(layers.Dense(128,activation ='relu',input_shape = (5,20))) #Dense layer, 4 neurons tanh activation - classification output

# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(32,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(16,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(8,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Flatten())
model.add(layers.Dense(3,activation='softmax'))#Dense layer, 4 neurons softmax activation - classification output
# model.summary()

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='val_loss', patience=10, verbose=2, min_delta = 1e-3)
history = model.fit(
    x_train, y_train,
    epochs=500, batch_size=32,
    validation_data=(x_test, y_test),
    verbose = 2,
    callbacks=[es]
    )
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
cv_score.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_score), np.std(cv_score)))
print(a)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# def metric(result,predict):
#     for idx,val in enumerate(result):

# model.save('bestmodel')
# model = keras.models.load_model('bestmodel')
# scores = model.evaluate(to_categorical(amino), to_categorical(label))

# make a prediction
data = pd.read_csv('protein-secondary-structure.test', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
amino_map = data.copy()
second_map = data.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
data.fillna(0,inplace = True)

total,label,data = re_formatdata(data)
amino_map.drop_duplicates(inplace = True)
second_map.drop_duplicates(inplace = True)
amino_map = amino_map.tolist()
second_map = second_map.tolist()

amino_map = map_int(amino_map)
second_map = map_int(second_map)


data = pd.DataFrame({'amino':total,'label':label})


# data.to_csv('data_format_train.csv',index = False)
# print(aaaa)
data['amino_count'] = data['amino'].apply(lambda x: len(x))

lenght = data['amino_count'].max()
for idx,rows in data.iterrows():
    data.at[idx,'amino'] = add_pading(rows['amino'],lenght)
    data.at[idx,'label'] = add_pading(rows['label'],lenght)

data = encoding_to_int(data,amino_map,second_map)
data.drop(['amino_count'],axis = 1, inplace = True)
amino = np.array(data['amino'].tolist())
label = np.array(data['label'].tolist())

# amino = to_categorical(amino)
amino = tf.one_hot(amino, depth = 20)


ynew = model.predict_classes(amino)
print(label)
print('------')
print(ynew)
print('-------')
# show the inputs and predicted outputs
acc = metrics_protein(label,ynew)
print(acc)

