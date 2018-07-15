from keras.layers import Conv2D, BatchNormalization, Activation, GlobalMaxPool2D


def merge_cnn(word_vec=None, kernel_size=1, filters=512):
    x = word_vec
    x = Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size], strides=[1, 1], padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = GlobalMaxPool2D()(x)

    return x


if __name__ == '__main__':
    from keras.layers import Dense, Input, Concatenate
    from keras.models import Model
    from keras.utils import plot_model

    filters = 256
    data_input = Input(shape=[28, 28, 1])
    x2 = merge_cnn(word_vec=data_input, kernel_size=2, filters=filters)
    x3 = merge_cnn(word_vec=data_input, kernel_size=3, filters=filters)
    x4 = merge_cnn(word_vec=data_input, kernel_size=4, filters=filters)
    x5 = merge_cnn(word_vec=data_input, kernel_size=5, filters=filters)

    x = Concatenate(axis=1)([x2, x3, x4, x5])
    x = BatchNormalization()(x)
    x = Dense(1000, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    model = Model(inputs=data_input, outputs=x)
    plot_model(model, './resnet.png', show_shapes=True)
