import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.layers import (LSTM, Dense, Flatten, GlobalAveragePooling1D, Input,
                            TimeDistributed)
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            print("Directory is already exists")
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
def build_model():
    # MODEL
    inputs = Input(shape=(3413, 32))
    x = Dense(128, activation='relu')(inputs)
    x = LSTM(32, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    #x = Flatten()(x)
    prediction = Dense(4, activation='softmax')(x)
    return Model(inputs=inputs, outputs=prediction)
    
classes = 4
emo = ['ang', 'neu', 'hap', 'sad']
speak = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05'] # cross validation
avg_acc = 0


savez = np.load("/home/gnlenfn/data/remote/pytorch_emotion/Feature/data_save.npz", allow_pickle=True)
k = np.where(savez['y'] == 'exc', 'hap', savez['y'])
# PAD DATASET
max_time = []
for x in savez['x']:
    tmp = x.shape[0]
    max_time.append(tmp)
max_len = max(max_time)
        
for sp in speak:
    # CREATE PATH
    model_path = "/home/gnlenfn/data/remote/pytorch_emotion/model/4emo/rnn/" + sp + "/"
    log_path = "/home/gnlenfn/data/remote/pytorch_emotion/log/4emo/rnn/"
    createFolder(model_path)
    createFolder(log_path)
    model_name = model_path + sp + '-{epoch:03d}.h5'
    
    # DATA SET UP
    le = preprocessing.LabelEncoder()
    idx_test  = np.where((savez['s'] == sp) & (savez['t'] == 'impro'))
    idx_train = np.where((savez['s'] != sp) & (savez['t'] == 'impro'))
        #train_set
    x_train  = savez['x'][idx_train]    
    x_train = sequence.pad_sequences(x_train, maxlen=max_len, dtype='float')
    y_train  = k[idx_train]
    y_train  = le.fit_transform(y_train)
    #print(y_train.shape)
        #test_set
    x_test   = savez['x'][idx_test]    
    x_test   = sequence.pad_sequences(x_test, maxlen=max_len, dtype='float')
    y_test   = k[idx_test]
    y_test   = le.fit_transform(y_test)
    #print(y_test.shape)
    
    # CALLBACKS
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    checkpoint  = keras.callbacks.ModelCheckpoint(model_name, verbose=0, monitor='val_accuracy',
                                                save_best_only=True, mode='auto')
    
    # TRAINING
    with tf.device("/gpu:0"):
        model = build_model()
        print(model.summary())
        model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])    
        
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), 
                            epochs=100, batch_size=512, callbacks=[tensorboard, checkpoint], verbose=2)
        
            #load model
        loaded_model = build_model()
        tmp = glob.glob(model_path + "*")
        tmp.sort()
        print(tmp[-1])
        loaded_model.load_weights(tmp[-1])
        loaded_model.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy')
                            
            #evaluate
        scores = loaded_model.evaluate(x_test, y_test, verbose=0)
        print("best model: {}: {:4f}%".format(loaded_model.metrics_names[1], scores[1]*100))
        avg_acc += scores[1] * 100
        
        # CONFUSION MATRIX
        Y_pred = model.predict(x_test)
        pred=np.argmax(Y_pred.reshape(x_test.shape[0],4),axis=1)
        print(confusion_matrix(y_test, pred))
        target_names = ['ang', 'hap', 'neu', 'sad']
        print(classification_report(y_test, pred, target_names=target_names))
    
print("Final Result: ", avg_acc / 5)
