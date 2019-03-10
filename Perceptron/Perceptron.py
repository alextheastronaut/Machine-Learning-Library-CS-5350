import numpy as np


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


def perceptron(data, num_epochs, learn_rate, **kwargs):
    # 0 = "standard" perceptron algorithm
    # 1 = Voted
    # 2 = Average
    percep_type = kwargs.get("percep_type", 0)

    # x = samples matrix
    # y = labels vector
    x, y = turn_data_to_x_and_y(data)

    m, n = data.shape
    w = np.zeros(n)

    if percep_type is not 0:
        # number of times current weight vector is correct.
        # the higher it is the more say its associated weight vector
        weight_count_tuples = list()
        times_w_correct = 0

    if percep_type is 2:
       combined_avg_vec = np.zeros(n)

    for t in range(num_epochs):

        if percep_type is 0:
            np.random.shuffle(x)

        for i in range(len(x)):
            # if wrong prediction, update weight vector
            if y[i] * np.dot(w, x[i]) <= 0:

                if percep_type is not 0:
                    weight_count_tuples.append((w, times_w_correct))
                    times_w_correct = 0

                w = w + learn_rate * y[i] * x[i]

            elif percep_type is not 0:
                times_w_correct += 1

            if percep_type is 2:
                combined_avg_vec += w

    if percep_type is 0:
        return w
    elif percep_type is 1:
        return weight_count_tuples
    else:
        return combined_avg_vec, weight_count_tuples


def find_error_on_test_data(percep_output, test_data):

    x, y = turn_data_to_x_and_y(test_data)
    times_wrong = 0

    for i in range(len(test_data)):
        if not isinstance(percep_output, list):
            predict = np.sign(np.dot(percep_output, x[i]))
        else:
            sum = 0
            for j in range(len(percep_output)):
                w_j, c_j = percep_output[j]
                sign = np.sign(np.dot(w_j, x[i]))
                sum += c_j * sign

            predict = np.sign(sum)

        if y[i] != predict:
            times_wrong += 1

    print(times_wrong / len(test_data))

if __name__ == "__main__":
    csv_train = "../DataSets/bank-note/train.csv"
    csv_test = "../DataSets/bank-note/test.csv"

    train_data = read_csv(csv_train)
    weight= perceptron(train_data, 10, 0.1, percep_type=0)
    #weight = weight / np.linalg.norm(weight)
    print(weight)
    #print(len(vec))

    test_data = read_csv(csv_test)
    find_error_on_test_data(weight, train_data)
    find_error_on_test_data(weight, test_data)