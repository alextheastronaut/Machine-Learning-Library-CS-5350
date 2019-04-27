import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
import matplotlib.pyplot as plt

def read_csv(CSV_file):
    data = list()
    with open(CSV_file, 'r') as f:
        for line in f:
            sample = line.strip().split(',')

            #Change label value from 0 to -1
            # if sample[len(sample) - 1] == '0':
            #     sample[len(sample) - 1] = -1

            data.append(sample)

    return np.array(data).astype(np.float)

def turn_data_to_x_and_y(data):

    m, n = data.shape

    # Last column vector of data is labels vector y
    # Everything else is x
    y = data[:, n - 1]
    x = data[:,: n - 1]

    b = np.array([[1] * len(x)])
    x = np.append(x, b.transpose(), axis=1)

    return x, y

def build_model(X, y, width, depth, epochs, select):
    model = Sequential()

    if select is 0:
        activ = 'relu'
        init = initializers.he_normal(seed=None)
    elif select is 1:
        activ = 'tanh'
        init = initializers.glorot_normal(seed=None)
    elif select is 2:
        activ = 'sigmoid'
        init = initializers.random_uniform(seed=None)


    model.add(Dense(width, input_dim=X.shape[1], activation=activ, kernel_initializer=init))
    for i in range(1, depth):
        model.add(Dense(width, activation=activ, kernel_initializer=init))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

    model.compile(optimizer='adam',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model.fit(X, y, epochs=epochs), model

def make_figure(t_loss, val_loss, widths, graph_title, x_label, y_label, picture_name):
    fig = plt.figure()
    plt.plot(widths, t_loss, color='blue', label='train')
    plt.plot(widths, val_loss, color='red', label='test')
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    fig.savefig(picture_name + '.png', dpi=100)
    plt.show()

if __name__ == '__main__':
    csv_train = "../DataSets/bank-note/train.csv"
    csv_test = "../DataSets/bank-note/test.csv"
    csv_testing = '../DataSets/test/testNN.csv'

    train_data = read_csv(csv_train)
    test_data = read_csv(csv_test)
    testing = read_csv(csv_testing)

    widths = [5, 10, 25, 50, 100]

    X_train, y_train = turn_data_to_x_and_y(train_data)
    X_test, y_test = turn_data_to_x_and_y(test_data)

    train_loss = [0] * 5
    test_loss = [0] * 5

    for i in range(len(widths)):
        train_history, model = build_model(X_train, y_train, widths[i], 3, 10, 1)
        train_loss[i] = train_history.history['loss'][9]
        test_loss[i] = model.evaluate(X_test, y_test, verbose=False)[0]

    print(train_loss)
    print(test_loss)
    make_figure(train_loss, test_loss, widths, 'tanh loss for depth 3', 'width', 'loss', 'tanh3')