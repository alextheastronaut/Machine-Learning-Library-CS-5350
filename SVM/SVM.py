import numpy as np
from scipy.optimize import minimize

def read_csv(CSV_file):
    data = list()
    with open(CSV_file, 'r') as f:
        for line in f:
            sample = line.strip().split(',')

            #Change label value from 0 to -1
            if sample[len(sample) - 1] == '0':
                sample[len(sample) - 1] = -1

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


def stochastic_sub_gradient_descent(data, gamma, d, C, epochs):

    sample_dim = data.shape[1]
    w = np.zeros(sample_dim)
    N = len(data)

    for t in range (epochs):
        np.random.shuffle(data)
        x, y = turn_data_to_x_and_y(data)

        if d == 0:
            r_t = gamma / (1 + t)
        else:
            r_t = gamma / (1 + gamma * t / d)

        for i in range(N):
            if y[i] * w.dot(x[i]) <= 1:
                w = w - r_t * w + r_t * N * C * y[i] * x[i]
            else:
                w_o = np.copy(w)
                w_o[len(w) - 1] = 0
                w = w - r_t * w_o

    return w


def print_error_for_different_C(training_data, test_data, gamma, d, C_arr, epochs):
    for C in C_arr:
        lrn_w = stochastic_sub_gradient_descent(training_data, gamma, d, C, epochs)

        train_error = get_w_error(lrn_w, training_data)
        test_error = get_w_error(lrn_w, test_data)

        print('C = ', C)
        print('Training Error: ', train_error)
        print('Test Error:, ', test_error)


def get_w_error(lrn_w, test_data):

    times_wrong = 0
    x, y = turn_data_to_x_and_y(test_data)

    for i in range(len(test_data)):
        if y[i] * lrn_w.dot(x[i]) < 0:
            times_wrong += 1

    return times_wrong / len(test_data)


if __name__ == '__main__':
    csv_train = "../DataSets/bank-note/train.csv"
    csv_test = "../DataSets/bank-note/test.csv"

    train_data = read_csv(csv_train)
    test_data = read_csv(csv_test)
    C1 = np.array([1, 10, 50, 100, 300, 500, 700])
    C1 = C1 / 873
    print_error_for_different_C(train_data, test_data, 0.1, 1, C1, 100)