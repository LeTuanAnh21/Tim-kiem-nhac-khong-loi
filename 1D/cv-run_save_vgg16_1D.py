import argparse
parser = argparse.ArgumentParser(description="run",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--datatype", default=2, type=int, help="datatype 1: 1D, 2:2D")
parser.add_argument("-d", "--lengthaudio", default=220499, type=int, help="length of audio")
parser.add_argument("-i", "--image_len", default=32, type=int, help="IMAGE_LEN")
parser.add_argument("-b", "--batch_size", default=32, type=int,  help="BATCH_SIZE")
parser.add_argument("-t", "--folder_train",default='train',  help="folder_train")
parser.add_argument("-r", "--folder_test", default='test',  help="folder test")
parser.add_argument("-e", "--num_epoch", default=3, type=int, help="epoch")
parser.add_argument("-p", "--patience_epoch", default=3, type=int, help="epoch patience")
parser.add_argument("-c", "--num_class", default=2, type=int, help="number of classes")
parser.add_argument("-l", "--learning_rate", default=0.0001, type=float, help="number of classes")
parser.add_argument("-k", "--nkfold", default=0, type=int, help="kfold ")
parser.add_argument("-f", "--folder", default='save_models', help="folder to save")
parser.add_argument("--seed", default=1,type=int, help="folder to save")
parser.add_argument("--mod", default='vgg', help="model")
parser.add_argument("--nfilter", default=64, type=int, help="number of filter (for cnn3)")

args = parser.parse_args()
IMAGE_LEN = args.image_len
BATCH_SIZE = args.batch_size
folder_train = args.folder_train
folder_test = args.folder_test
path_2 = folder_train + folder_test
path_named = path_2.replace("/", ".")
num_epoch = args.num_epoch
num_class = args.num_class
learning_rate=args.learning_rate
patience_epoch=args.patience_epoch
nkfold=args.nkfold
folder_save = args.folder
seed_v = args.seed
mod = args.mod

import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
#khai báo frame work tensorflow
import tensorflow as tf
import random
#import keras từ frame work tensorflow
from tensorflow import keras
import tensorflow.keras 
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, MaxPool2D
from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.layers import MaxPooling2D

import pandas as pd
# gan cac seed de co kq thuc nghiem giong
tf.random.set_seed(seed_v)
random.seed(seed_v)
np.random.seed(seed_v)

import time
start = time.time()
## tim tap tin de tranh bi trung
import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# Hàm để load dữ liệu từ các thư mục con và chuyển thành numpy array
# nthai: add label: chi su dung khi class trong train la so class trong test
def load_data_from_folder(folder_path, len_audio = 110249):
    data = []
    label = []
    i_label = -1
    # Duyệt qua tất cả các thư mục con
    for root, dirs, files in os.walk(folder_path):
        # print('root=', root)
        for filename in files:
            # print('filename=',filename)
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                data1 = pd.read_csv(file_path, sep=',') 
                if len(data1) == len_audio:
                    data.append(data1)
                    label.append(i_label)
        i_label = i_label + 1
    return np.array(data), label, i_label  # Convert the combined data to a numpy array

if args.datatype == 1:
    # print('1D loading data')
    # Load dữ liệu huấn luyện và kiểm tra
    train_ds, y_train, i_label = load_data_from_folder(folder_train, len_audio = args.lengthaudio)
    test_ds, y_test, i_label = load_data_from_folder(folder_test, len_audio = args.lengthaudio)

    # print('train_ds',train_ds.shape)
    # print('test_ds',test_ds.shape)
    # print('y_train',y_train)
    # print('y_test',y_test)

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing

def model_effi(num_classes = num_class,   image_size = IMAGE_LEN, batch_size = BATCH_SIZE):
    img_input = Input(shape=(image_size,image_size,3))
    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    x = Flatten(name='flatten')(x)
    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    # Logits layer
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model

if nkfold in [0,1]:
    if mod == 'vgg':
        name_model = ''
    else:
        name_model = mod + '_'
    name_saved= name_model + path_named + '_c'+ str(num_class)+ '_s' + str(IMAGE_LEN) + '_b'+ str(BATCH_SIZE) +  '_e'+ str(num_epoch) +'_p'+ str(patience_epoch) +  '_lr'+str(learning_rate) + '_se'+ str(seed_v) +'_k'+ str(nkfold) + 'nfilter'+str(args.nfilter) 
    print ('name_saved='+name_saved)
    n_files = find(name_saved + '*.json',folder_save )
    if len(n_files)>0:
        print('thuc nghiem '+name_saved+' da lam roi10!')
        exit()
   
    elif mod == 'fc':
        from tensorflow.keras.callbacks import EarlyStopping
        model = Sequential()       
        model.add(InputLayer(input_shape=(args.lengthaudio,)))  
        model.add(Dense(units=i_label, activation="softmax"))
        opt = Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        print('fc')
        model.summary()
  
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epoch)   
    # nthai, chuyen label dang categorical voi so phan lop la i_label
    
    print('i_label===',i_label)
    y_train = keras.utils.to_categorical(y_train,  num_classes = i_label )
    y_test = keras.utils.to_categorical(y_test,  num_classes = i_label )

    print(model.summary())
    print('before learning====================')
    print('train_ds.shape')
    print(train_ds.shape)
    print(y_train)

    print('test_ds.shape')
    print(test_ds.shape)
    print(y_test)

    history = model.fit(train_ds,y_train,validation_data=(test_ds,y_test), epochs=num_epoch, verbose=1,callbacks=[callback])
    from datetime import datetime
    now=datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")

    name_saved=  name_saved +'_' + str(date_time) #+'_k'+str(i)

    print('Model Saved!')
    model.save(folder_save+'/'+ name_saved +'.keras')
    model_json = model.to_json()
    with open( folder_save + '/' + name_saved +".json", "w") as json_file:
        json_file.write(model_json)
        
        
        # luu lai log
    end = time.time()
    print("time run: ", end - start)
    ep_arr=range(1, len(history.history['accuracy'])+1, 1)
    idx = len(history.history['accuracy'])-1 
    train_acc = history.history['accuracy']
    val_acc= history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    title_cols = np.array(["ep","train_acc","valid_acc","train_loss","valid_loss"])      
    res = (ep_arr, train_acc, val_acc, train_loss, val_loss)
    res = np.transpose(res)
    combined_res = np.array(np.vstack((title_cols, res)))
    
    log_name1 = name_saved +'s1'
    np.savetxt(folder_save + '/'+log_name1 +".txt", combined_res, fmt="%s",delimiter="\t") 

    #log 2 luu lai tham so va cac thong tin ve sample
    log_name2 = name_saved +'s2_'  + 'time'+ str(round(end - start,2)) + 'acc' +str(round(val_acc[idx],3))
    
    
    with open(folder_save + '/'+log_name2+ ".txt", 'w') as f:f.write(str(args))
    title_cols = np.array(["samples_train","samples_test","train_acc","train_loss","val_acc","val_loss"])  