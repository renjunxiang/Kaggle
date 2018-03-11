import pandas as pd
import numpy as np
from keras.models import Model
from keras.optimizers import SGD, Adagrad, Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape, BatchNormalization, Dropout, \
    GaussianNoise
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# 设置显存
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# 读取数据
train_data = pd.read_csv('./data/train.csv')  # 42000
test_data = pd.read_csv('./data/test.csv')

train_x = train_data.iloc[:, 1:] / 256
train_x = np.array(train_x)
train_y = train_data.iloc[:, 0:1]
train_y = pd.get_dummies(train_y.astype(str))
train_y = np.array(train_y)

test_x = np.array(test_data) / 256
train_train_x, train_test_x, train_train_y, train_test_y = train_test_split(train_x, train_y, test_size=0.2)
#####################################################################################################################
# kaggle准确率到达99.3%
input_data = Input(shape=[784])
x = GaussianNoise(stddev=0.2)(input_data) #加入噪音
x = Reshape(target_shape=[28, 28, 1])(input_data)
x = Conv2D(filters=20, kernel_size=[3, 3], padding='same', activation='relu', data_format='channels_last')(x)
x = BatchNormalization(epsilon=0.000001, axis=1)(x) #标准化输出
x = MaxPooling2D(pool_size=[2, 2], strides=2, padding='valid')(x)
x = Conv2D(filters=50, kernel_size=[3, 3], padding='same', activation='relu', data_format='channels_last')(x)
x = BatchNormalization(epsilon=0.000001, axis=1)(x)
x = MaxPooling2D(pool_size=[2, 2], strides=2, padding='valid')(x)
x = Conv2D(filters=50, kernel_size=[2, 2], padding='same', activation='relu', data_format='channels_last')(x)
x = BatchNormalization(epsilon=0.000001, axis=1)(x)
x = MaxPooling2D(pool_size=[2, 2], strides=2, padding='valid')(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(units=500, activation='relu')(x)
x = Dense(units=128, activation='relu')(x)
x = Dense(units=10, activation='softmax')(x)

model = Model(inputs=input_data, outputs=x)
optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#####################################################################################################################
model.fit(x=train_train_x, y=train_train_y, batch_size=500, epochs=1,
          validation_data=[train_test_x, train_test_y], verbose=2)
model.save('./models/model_BatchNormalization_noise1.model')

# from keras.models import load_model
# model=load_model('./models/model_BatchNormalization_noise1.model')
test_y = model.predict(x=test_x)
test_y_label = pd.DataFrame(test_y)
test_y_label = test_y_label.agg(lambda x: np.argmax(x), axis=1)
pd.DataFrame({'ImageId': range(1, 1 + len(test_y_label)), 'Label': test_y_label},
             columns=['ImageId', 'Label']).to_csv('./result/result_noise.csv', index=False)



# plt.imshow(test_x[0].reshape(28, 28), cmap='gray')
