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
        delta_b = delta_b + y[i] - np.dot(w, x[i]) - b

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
        #print("hi", np.linalg.norm(w2 - w1))
        if np.linalg.norm(w2 - w1) <= threshold:
            cost_at_t.append(cost_function(w2, actual_y, x, b))
            return w2, b, cost_at_t

        w1 = w2

    cost_at_t.append(cost_function(w2, actual_y, x, b))
    return w2, b, cost_at_t


def random_stochastic_gradient_descent(data, attributes, r, threshold, num_iter):
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


def stochastic_gradient_descent(data, attributes, r, threshold, num_iter):
    w = np.array([0] * len(attributes)).astype(float)
    b = 0
    actual_y = data[:, len(w)]
    x = data[:, :len(w)]

    for z in range(num_iter):
        for i in range(len(data)):
            err = (actual_y[i] - (np.dot(w, x[i]) + b))

            for j in range(len(w)):
                w[j] += r * err * x[i][j]

            b += r * err

            print(w)
            print(b)


def cost_function(w, y, x, b):
    cost_sum = 0
    for i in range(len(x)):
        cost_sum += (y[i] - np.dot(w, x[i]) - b) ** 2

    return 0.5 * cost_sum


def cost_function_data(w, b, data):
    y = data[:, len(w)]
    x = data[:, :len(w)]
    return cost_function(w, y, x, b)


def test_gradient(w, b, data):
    actual_y = data[:, len(w)]
    x = data[:, :len(w)]
    grad, delta_b = gradient(w, actual_y, x, b)
    print(grad)
    print(b - delta_b)


def calc_optimal_w(data, attributes):
    y = data[:, len(attributes)]
    x = data[:, :len(attributes)]
    b = np.array([[1] * len(data)])
    x = np.concatenate((x, b.transpose()), axis=1)
    m,n = np.shape(x)
    x_x = np.linalg.inv(np.dot(np.transpose(x), x))
    x_y = np.dot(y.reshape(1, m), x)
    return np.dot(x_y, x_x)


def make_figure(x, y, graph_title, x_label, y_label, picture_name):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #path = "/home/alex/MachineLearning/pics/"
    fig.savefig(picture_name + '.png', dpi=100)
    plt.show()


def plot_descent(descent_method, r, threshold, num_iter, training_data, test_data, atts, plot_title, x_label, y_label, pic_name):
    w, b, cost_at_iter_t = descent_method(training_data, atts, r, threshold, num_iter)
    make_figure([x for x in range(1, len(cost_at_iter_t) + 1)], cost_at_iter_t, plot_title, x_label, y_label, pic_name)
    optimal = calc_optimal_w(training_data, atts)
    print(w)
    print(b)
    print("Cost: ", cost_function_data(w, b, test_data))
    print(optimal)
    print(np.linalg.norm(optimal - w))


if __name__ == "__main__":
    #training_data = read_csv("../DataSets/concrete/train.csv")
    #test_data = read_csv("../DataSets/concrete/test.csv")
    #atts = read_txt("../DataSets/concrete/data-desc-readable.txt")
    #plot_descent(batch_gradient_descent, .014, 10 ** -6, 50, training_data, test_data, atts, "hi", "Number of Iterations", "Cost Function", "HW2test")

    data = read_csv("../DataSets/LMStest/matrix.csv")
    atts = read_txt("../DataSets/LMStest/var.txt")
    stochastic_gradient_descent(data, atts, 0.1, 0, 1)
    print(calc_optimal_w(data, atts))
    #w_s, cost_s = stochastic_gradient_descent(training_data, atts, 1 / 2 ** 7, 1 * 10 ** -7)
    #make_figure([x for x in range(1, len(cost_s) + 1)], cost_s, "Stochastic Gradient Descent for r = 2^(-9)", "Number of Iterations",
    #            "Cost Function", "HW2sgd")
    #print(w_s)
    #print("COST: ", cost_function_data(w_s, test_data))
