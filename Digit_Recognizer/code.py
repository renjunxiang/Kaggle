import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, initializers, Flatten, Dropout, Reshape
from keras.layers import Conv2D, InputLayer
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adam
from keras import regularizers
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./data/train.csv')  # 42000
test_data = pd.read_csv('./data/test.csv')

train_x = train_data.iloc[:, 1:]/256
train_x = np.array(train_x)
train_y = train_data.iloc[:, 0:1]
train_y = pd.get_dummies(train_y.astype(str))
train_y = np.array(train_y)

test_x=np.array(test_data)/256

train_train_x, train_test_x, train_train_y, train_test_y = train_test_split(train_x, train_y, test_size=0.3)

model = Sequential()
model.add(Reshape(input_shape=[784],
                  target_shape=[28, 28, 1],
                  name='Reshape_2d'))
model.add(Conv2D(filters=16,
                 kernel_size=[3, 3],
                 padding='same',
                 activation='relu',
                 data_format='channels_last'))
model.add(MaxPooling2D(pool_size=[2, 2],
                       strides=2,
                       padding='valid'))
model.add(Conv2D(filters=64,
                 kernel_size=[3, 3],
                 padding='same',
                 activation='relu',
                 data_format='channels_last'))
model.add(MaxPooling2D(pool_size=[2, 2],
                       strides=2,
                       padding='valid'))
model.add(Flatten(name='Flatten'))
model.add(Dense(units=64,
                activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10,
                activation='softmax'))
optimizer = Adagrad(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_train_x, y=train_train_y, batch_size=500, epochs=5,
          validation_data=[train_test_x, train_test_y], verbose=1)
model.save('./models/model.model')
test_y=model.predict(x=test_x)
test_y_label=pd.DataFrame(test_y)
test_y_label=test_y_label.agg(lambda x:np.argmax(x),axis=1)
pd.DataFrame({'ImageId':range(1,1+len(test_y_label)),'Label':test_y_label},
             columns=['ImageId','Label']).to_csv('./result/result.csv',index=False)



plt.imshow(test_x[0].reshape(28, 28), cmap='gray')



