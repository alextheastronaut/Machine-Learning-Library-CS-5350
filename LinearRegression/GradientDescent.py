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


def gradient(w, y, x, b):
    grad = np.zeros(len(w))
    delta_b = 0
    for i in range(len(x)):
        grad = grad - (y[i] - np.dot(w, x[i]) - b) * x[i]
        delta_b = delta_b  + y[i] - np.dot(w, x[i]) - b

    #print(np.linalg.norm(grad))
    return grad, delta_b


def batch_gradient_descent(data, attributes, r, threshold, num_iter):
    w1 = np.array([0] * len(attributes))
    b = 0
    actual_y = data[:, len(w1)]
    x = data[:, :len(w1)]
    cost_at_t = list()
    for z in range(num_iter):
        cost_at_t.append(cost_function(w1, actual_y, x, b))
        #print(cost_function(w1, actual_y, x, b))
        grad_vec, delta_b = gradient(w1, actual_y, x, b)
        w2 = w1 - r * grad_vec
        b = b + r * delta_b
        print(np.linalg.norm(w2 - w1))
        if np.linalg.norm(w2 - w1) <= threshold:
            cost_at_t.append(cost_function(w2, actual_y, x, b))
            return w2, b, cost_at_t

        w1 = w2

    cost_at_t.append(cost_function(w2, actual_y, x, b))
    return w2, b, cost_at_t


def stochastic_gradient_descent(data, attributes, r, threshold, num_iter):
    w = np.array([0] * len(attributes)).astype(float)
    b = 0
    actual_y = data[:, len(w)]
    x = data[:, :len(w)]
    cost_at_t = list()

    for z in range(num_iter):
        rand_idx = np.random.randint(0, len(x))
        ran_sample = x[rand_idx]
        err = actual_y[rand_idx] - np.dot(w, ran_sample) - b
        cost_at_t.append(cost_function(w, actual_y, x, b))
        w_prev = np.copy(w)

        for j in range(len(w)):
            w[j] += r * err * ran_sample[j]

        b += r * err

        print(np.linalg.norm(w - w_prev))

    cost_at_t.append(cost_function(w, actual_y, x, b))
    return w, b, cost_at_t


def cost_function(w, y, x, b):
    cost_sum = 0
    for i in range(len(x)):
        cost_sum += (y[i] - np.dot(w, x[i]) - b) ** 2

    return 0.5 * cost_sum


def cost_function_data(w, b, data):
    y = data[:, len(w)]
    x = data[:, :len(w)]
    return cost_function(w, y, x, b)


def calc_optimal_w(data, attributes):
    y = data[:, len(attributes)]
    x = data[:, :len(attributes)]
    m,n = np.shape(x)
    x_x = np.linalg.inv(np.dot(np.transpose(x), x))
    l,k = np.shape(x_x)
    x_y = np.dot(y.reshape(1, m), x)
    return np.dot(x_y, x_x)
        #np.dot(np.linalg.inv(np.dot(x, np.transpose(x))), np.dot(x, y))


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
    optimal = calc_optimal_w(training_data, atts)
    descent_method = stochastic_gradient_descent
    w, b, cost_at_iter_t = descent_method(training_data, atts, 1 / 2 ** 10, 1 * 10 ** -6, 50000)
    make_figure([x for x in range(1, len(cost_at_iter_t) + 1)], cost_at_iter_t,
                "Stochastic Gradient Descent for r = 2^(-10)",
                "Number of Iterations", "Cost Function", "HW2sgd")
    print(w)
    print(b)
    print("Cost: ", cost_function_data(w, b, test_data))
    print(np.linalg.norm(optimal - w))
    #w_s, cost_s = stochastic_gradient_descent(training_data, atts, 1 / 2 ** 7, 1 * 10 ** -7)
    #make_figure([x for x in range(1, len(cost_s) + 1)], cost_s, "Stochastic Gradient Descent for r = 2^(-9)", "Number of Iterations",
    #            "Cost Function", "HW2sgd")
    #print(w_s)
    #print("COST: ", cost_function_data(w_s, test_data))
