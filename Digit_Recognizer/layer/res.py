from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.layers import GlobalMaxPool2D, Dropout, Conv2D, BatchNormalization, Activation, Add
from keras.utils import plot_model


def block(x, filters=256, kernel_size=3):
    x_Conv_1 = Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding='same')(x)
    x_Conv_1 = BatchNormalization()(x_Conv_1)
    x_Conv_1 = Activation(activation='relu')(x_Conv_1)
    x_Conv_2 = Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding='same')(x_Conv_1)
    x_Conv_2 = Add()([x, x_Conv_2])
    x_Conv_2 = BatchNormalization()(x_Conv_2)
    x = Activation(activation='relu')(x_Conv_2)
    return x


if __name__ == '__main__':
    kernel_size = [3, 3]
    DIM = 512

    data_input = Input(shape=[28, 28, 1])
    block1 = block(x=data_input, filters=256, kernel_size=3)
    block2 = block(x=block1, filters=256, kernel_size=3)
    x = GlobalMaxPool2D()(block2)
    x = BatchNormalization()(block2)
    x = Dense(1000, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(202, activation="sigmoid")(x)
    model = Model(inputs=data_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    plot_model(model, './resnet.png', show_shapes=True)
