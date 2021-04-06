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
# pre-processing dataset
pd.options.display.max_colwidth
np.set_printoptions(threshold=np.inf)
data = pd.read_csv('protein-secondary-structure.train', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
amino_map = data.copy()
second_map = data.copy()
amino_map.dropna(inplace = True)
second_map.dropna(inplace = True)
amino_map = amino_map['amino']
second_map = second_map['label']
data.fillna(0,inplace = True)

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

def map_int(ls):
    amino_dict = {}
    code = 0
    for i in ls:
        amino_dict[i] = code
        code += 1

    return amino_dict


def encoding_to_int(df,first_map,second_map):
    for idx,rows in df.iterrows():
        if idx == 0:
            print(rows)
        row_encode = []
        for code in rows['amino']:
            if code != 0:
                row_encode.append(first_map.get(code))
            else :
                print('!!!!!!!!!1')
                row_encode.append(20)
        if idx == 0:
            print(row_encode)
        df.at[idx,"amino"] = row_encode
        row_encode = []
        for code in rows['label']:
            if code != 0:
                row_encode.append(second_map.get(code))
            else :
                row_encode.append(20)
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
# padding sequences

for idx,rows in data.iterrows():
    data.at[idx,'amino'] = add_pading(rows['amino'],lenght)
    data.at[idx,'label'] = add_pading(rows['label'],lenght)

print(data['amino'])
print('!++!++!+!+!+!')
data = encoding_to_int(data,amino_map,second_map)
data.drop(['amino_count'],axis = 1, inplace = True)
amino = np.array(data['amino'].to_list())
label = np.array(data['label'].to_list())

# amino = keras.preprocessing.sequence.pad_sequences(amino)
# label = keras.preprocessing.sequence.pad_sequences(label)
# print(amino)
x_train, x_test, y_train, y_test = train_test_split(amino, label, test_size=0.2)

# print(a)
# x_train = to_categorical(x_train)
# x_test = to_categorical(x_test)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
x_train = tf.one_hot(x_train, depth =20)
x_test = tf.one_hot(x_test, depth =20)
y_train = tf.one_hot(y_train, depth =3)
y_test = tf.one_hot(y_test, depth =3)

# print(x_train)
# print(x_train.shape)
# label = to_categorical(label)
# amino = to_categorical(amino)
amino = tf.one_hot(amino, depth = 20)
label1 = label.copy()
# print(amino.shape)
# print(label)
# print('==============')
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# # print('===')
# # print(aaaa)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv_score = []
tscv = TimeSeriesSplit(n_splits=5)

# for train,test in tscv.split(amino,label):
    # model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


model = keras.Sequential()
# model.add(layers.Dense(128, input_shape = (498,21) , activation ='relu'))
# model.add(layers.Dense(8,activation ='relu'))
model.add(layers.Masking(mask_value=0., input_shape=(498, 20)))
# model.add(layers.LSTM(128,return_sequences = True))#recurrent layer , 128 neurons
# model.add(layers.Bidirectional(layers.GRU(64,return_sequences=True, activation = 'relu')))#recurrent layer 1, 64 neurons
# model.add(layers.Bidirectional(layers.GRU(32, return_sequences=True, activation = 'relu'))) #recurrent layer 2, 32 neurons
# model.add(layers.Bidirectional(layers.GRU(8,return_sequences=True))) #recurrent layer 3, 16 neurons
model.add(layers.Dense(128,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output

# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(64,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(32,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(16,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(8,activation ='relu')) #Dense layer, 4 neurons tanh activation - classification output
model.add(layers.Dropout(0.8))
# model.add(layers.Flatten())
model.add(layers.Dense(3,activation='softmax'))#Dense layer, 4 neurons softmax activation - classification output
model.add(layers.Activation('softmax'))
model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

model.summary()

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
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

# # make a prediction
# data = pd.read_csv('protein-secondary-structure.test', skiprows = 9, delim_whitespace = True, header = None, names =['amino','label'])
# amino_map = data.copy()
# second_map = data.copy()
# amino_map.dropna(inplace = True)
# second_map.dropna(inplace = True)
# amino_map = amino_map['amino']
# second_map = second_map['label']
# data.fillna(0,inplace = True)

# total,label,data = re_formatdata(data)
# amino_map.drop_duplicates(inplace = True)
# second_map.drop_duplicates(inplace = True)
# amino_map = amino_map.tolist()
# second_map = second_map.tolist()

# amino_map = map_int(amino_map)
# second_map = map_int(second_map)


# data = pd.DataFrame({'amino':total,'label':label})


# # data.to_csv('data_format_train.csv',index = False)
# # print(aaaa)
# data['amino_count'] = data['amino'].apply(lambda x: len(x))

# lenght = data['amino_count'].max()
# for idx,rows in data.iterrows():
#     data.at[idx,'amino'] = add_pading(rows['amino'],lenght)
#     # data.at[idx,'label'] = add_pading(rows['label'],lenght)

# data = encoding_to_int(data,amino_map,second_map)
# data.drop(['amino_count'],axis = 1, inplace = True)
# amino = np.array(data['amino'].tolist())
# label = np.array(data['label'].tolist())

# amino = to_categorical(amino)


ynew = model.predict_classes(amino)
print(label1[88])
print('------')
print(ynew[0])
print('-------')
# show the inputs and predicted outputs
acc = metrics_protein(label1[88:],ynew)
print(acc)

