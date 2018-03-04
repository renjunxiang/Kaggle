import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout,Reshape
from keras.layers import Conv2D, InputLayer
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPool2D
from keras.optimizers import SGD, Adagrad, Adam

train_data = pd.read_csv('./data/train.csv') #42000
test_data = pd.read_csv('./data/test.csv')

train_x=train_data.iloc[:,1:]
train_y=train_data.iloc[:,0:1]
train_y=pd.get_dummies(train_y.astype(str))

model = Sequential()
model.add(InputLayer(input_shape=[784]))
model.add(Reshape(input_shape=[784], target_shape=[28, 28, 1], name='Reshape_2d'))
model.add(Conv2D(filters=16,kernel_size=[3,3],padding='same'))
model.add(MaxPool2D(pool_size=[2,2],strides=[1,1],padding='valid'))
model.add(Conv2D(filters=64,kernel_size=[3,3],padding='same'))
model.add(MaxPool2D(pool_size=[2,2],strides=[1,1],padding='valid'))
model.add(Flatten(name='Flatten'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
optimizer = Adagrad(lr=0.003)
model.compile(optimizer=optimizer,loss='categorical_crossentropy')
model.fit(x=np.array(train_x),y=np.array(train_y),batch_size=200,epochs=2,validation_split=0.2)



