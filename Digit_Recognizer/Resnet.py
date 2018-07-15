import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Reshape, GaussianNoise, BatchNormalization, GlobalMaxPool2D, Dropout
from layer import block
from evaluate import accu_score, f1_avg, p2tag
import time

# 读取数据
train_data = pd.read_csv('./data/train.csv')  # 42000
test_data = pd.read_csv('./data/test.csv')

train_x = train_data.iloc[:, 1:] / 256
train_x = np.array(train_x)
train_y = train_data.iloc[:, 0:1]
train_y = pd.get_dummies(train_y.astype(str))
train_y = np.array(train_y)

test_x = np.array(test_data) / 256
data_train_x, data_valid_x, data_train_y, data_valid_y = train_test_split(train_x, train_y,
                                                                          test_size=0.2, random_state=1)
###############################################################################################################
filters = 256
batch_size = 256

input_data = Input(shape=[784])
x = GaussianNoise(stddev=0.2)(input_data)  # 加入噪音
input_reshape = Reshape(target_shape=[28, 28, 1])(input_data)

block1 = block(x=input_reshape, filters=filters, kernel_size=3)
block2 = block(x=block1, filters=filters, kernel_size=3)
x = GlobalMaxPool2D()(block2)
x = Dense(1000, activation="relu")(x)
x = BatchNormalization()(x)
x = Dense(10, activation="softmax")(x)
model = Model(inputs=input_data, outputs=x)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

n_start = 1
n_end = 51
score_list = []
for i in range(n_start, n_end):
    model.fit(x=data_train_x, y=data_train_y, batch_size=batch_size, epochs=1, verbose=2)

    model.save('./models/Resnet_filters_%d_bs_%d_epochs_%d.h5' % (filters, batch_size, i))

    y1 = model.predict(data_valid_x)
    y1_tag = p2tag(y1)
    valid_tag = p2tag(data_valid_y)
    accu = accu_score(y1_tag, valid_tag)
    f1 = f1_avg(y1_tag, valid_tag)
    score_list.append([i, accu, f1])
    metrics = pd.DataFrame(score_list, columns=['batch_size', 'accu', 'f1'])
    print(metrics)
    metrics.to_csv('./metrics/metrics_Resnet_filters_%d_bs_%d.csv' % (filters, batch_size),
                   index=False)

    y2 = model.predict(test_x)
    y2_tag = p2tag(y2)
    result = pd.DataFrame({'ImageId': range(1, 1 + len(y2_tag)), 'Label': y2_tag},
                          columns=['ImageId', 'Label'])
    result.to_csv('./result/result_Resnet_filters_%d_bs_%d_epochs_%d.csv' % (filters, batch_size, i),
                  index=False)

print('end', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# nohup python CNN_merge.py 2>&1 &
