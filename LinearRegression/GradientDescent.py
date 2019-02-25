import numpy as np
import matplotlib.pyplot as plt


def read_csv(CSV_file):
    data = list()
    with open(CSV_file, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    return np.array(data).astype(np.float)


def read_txt(txt_file):
    atts = list()
    with open(txt_file, 'r') as f:
        for line in f:
            atts.append(line.strip())

    return atts


def batch_gradient_descent(data, attributes, r, threshold):
    w1 = np.array([0] * len(attributes))
    actual_y = data[:, len(w1)]
    x = data[:, :len(w1)]
    cost_at_t = list()
    while True:
        val = cost_function(w1, actual_y, x)
        print(val)
        cost_at_t.append(val)
        w2 = w1 - r * gradient(w1, actual_y, x)
        #print(np.linalg.norm(w2 - w1))
        if np.linalg.norm(w2 - w1) <= threshold:
            print("Cost: ", cost_function(w2, actual_y, x))
            return w2, cost_at_t

        w1 = w2


#def stochastic_gradient_descent():


def cost_function(w, y, x):
    cost_sum = 0
    for i in range(len(w)):
        cost_sum += (y[i] - np.dot(w, x[i])) ** 2

    return 0.5 * cost_sum


def gradient(w, y, x):
    grad = [0] * len(w)
    for j in range(len(w)):
        for i in range(len(x)):
            grad[j] -= (y[i] - np.dot(w, x[i])) * x[i][j]

    return np.array(grad)


def make_figure(x, y, graph_title, x_label, y_label, picture_name):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    path = "/home/alex/MachineLearning/pics/"
    fig.savefig(path + picture_name + '.png', dpi=100)
    plt.show()


if __name__ == "__main__":
    training_data = read_csv("../DataSets/concrete/train.csv")
    test_data = read_csv("../DataSets/concrete/test.csv")
    atts = read_txt("../DataSets/concrete/data-desc-readable.txt")
    w, cost_at_iter_t = batch_gradient_descent(training_data, atts, 1 / 2 ** 7, 1 * 10 ** -6)
    make_figure([x for x in range(1, len(cost_at_iter_t) + 1)], cost_at_iter_t,
                "Batch Gradient Descent for r = 2^(-7)",
                "Number of Iterations", "Cost Function", "HW2bgd")
