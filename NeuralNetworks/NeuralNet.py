import numpy as np

class NeuralNet:
    def __init__(self, width, train_data, **kwargs):

        self.width = width

        # Separate attributes from labels
        n, d = train_data.shape
        self.y = train_data[:, d - 1]
        self.X = train_data[:, : d - 1]
        b = np.array([[1] * n])
        # Prepend bias column to training matrix
        self.X = np.concatenate((b.transpose(), self.X), axis=1)

        self.weights = kwargs.get('weights', init_weight_arr(width, d))
        self.nodes = init_nodes_arr(width, d)


    def predict(self, x_i):
        self.nodes[0] = x_i
        self.update_sigmoid_layer(1, len(x_i))
        self.update_sigmoid_layer(2, self.width)

        return self.weights[2][0].dot(self.nodes[2])

    def update_sigmoid_layer(self, layer_idx, num_nodes):
        for node_idx in range(1, num_nodes):
            lin_comb = self.weights[layer_idx - 1][node_idx].dot(self.nodes[layer_idx - 1])
            self.nodes[layer_idx][node_idx] = sigmoid(lin_comb)

    def stoch_grad_descent(self, gamma, d, epochs):
        sample_idx = [x for x in range(len(self.X))]
        for t in range(epochs):
            np.random.shuffle(sample_idx)
            for i in sample_idx:
                x_i = self.X[i]
                y = self.y[i]
                r = learn_rate(gamma, d, t)
                grad = self.back_prop(x_i, y)
                update_weights(self.weights, r, grad)

    def back_prop(self, x_i, y):
        cache = init_cache_arr(self.width)
        grad = init_grad_arr(self.width, len(x_i))

        # Cache y and calc gradient in last hidden layer
        # print(cross_entropy(y, sigmoid(self.predict(x_i))))
        # print(sigmoid_prime(self.predict(x_i)))
        # print(cross_entropy_deriv(y, self.predict(x_i)) * sigmoid_prime(self.predict(x_i)))
        # print()
        cache[2][0] = cross_entropy_deriv(y, sigmoid(self.predict(x_i)) * sigmoid_prime(self.predict(x_i)))
        for n in range(self.width):
            grad[2][0][n] = cache[2][0] * self.nodes[2][n]

        # back propogate 2nd then 1st hidden layer
        self.back_prop_layer(2, self.width, self.width, cache, grad)
        self.back_prop_layer(1, len(x_i), self.width, cache, grad)

        # print(cache)
        # print()
        # for two_d in grad:
        #     print('layer')
        #     print(two_d)
        # print(len(grad))

        return grad

    def back_prop_layer(self, layer, num_nodes_0, num_nodes_1, cache, grad):
        """
        Updates cache and gradient of hidden layer in NN
        :param layer: index of layer
        :param num_nodes_0 number of nodes in layer - 1
        :param num_nodes_1: number of nodes in layer
        :param cache: the cache
        :param grad: the gradient
        :return:
        """

        #Iterate through each node, except the bias, in the layer
        for node_1 in range(1, num_nodes_1):
            # sigmoid * (1 - sigmoid)
            sig_prime = sigmoid_prime(self.nodes[layer][node_1])

            #Populate cache for layer

            # Vector of outgoing edges from node_1
            outgoing_edges = self.weights[layer][:, node_1]
            delta = cache[layer].dot(outgoing_edges)
            cache[layer - 1][node_1] = delta * sig_prime
            # print(delta * sig_prime)

            # For each ingoing edge to node_1
            for edge in range(num_nodes_0):
                grad[layer - 1][node_1][edge] = delta * sig_prime * self.nodes[layer - 1][edge]


                #cache_res = 0
                #for node_1 in range(1, num_nodes_1):
                #    cache_res += cache_res[layer_idx + 1][node_1]

def cross_entropy(y, y_t):
    return y * np.log2(y_t) + (1 - y) * np.log2(1 - y_t)


def cross_entropy_deriv(y, y_t):
    return y_t - 1 if y == 1 else y_t


def update_weights(weights, r, grad):
    for layer in range(len(weights)):
        weights[layer] = weights[layer] - r * grad[layer]


def init_nodes_arr(width, d):
    nodes = list()
    nodes.append(np.zeros(d)) # input layer
    nodes.append(np.zeros(width)) # hidden layer 1
    nodes.append(np.zeros(width)) # hidden layer 2
    nodes[1][0] = nodes[2][0] = 1 # hidden layer bias

    return nodes


def init_cache_arr(width):
    cache = list()
    cache.append(np.zeros(width)) # hidden layer 1
    cache.append(np.zeros(width)) # hidden layer 2
    cache.append(np.zeros(1)) # output

    return cache


def init_weight_arr(width, d):
    weights = list()
    weights.append(np.random.rand(width, d)) #layer 1
    weights.append(np.random.rand(width, width)) #layer 2
    weights.append(np.random.rand(1, width)) #output

    return weights


def init_grad_arr(width, d):
    grad = list()
    grad.append(np.zeros((width, d)))
    grad.append(np.zeros((width, width)))
    grad.append(np.zeros((1, width)))

    return grad

def sigmoid(x):
    x = np.float128(x)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(z):
    return z * (1.0 - z)


def learn_rate(gamma, d, t):
    return gamma / (1.0 + gamma * t / d)


def turn_data_to_x_and_y(data):

    m, n = data.shape

    # Last column vector of data is labels vector y
    # Everything else is x
    y = data[:, n - 1]
    x = data[:,: n - 1]

    b = np.array([[1] * len(x)])
    x = np.append(x, b.transpose(), axis=1)

    return x, y


def NN_error(NN, data):
    X, y = turn_data_to_x_and_y(data)
    times_wrong = 0

    for i in range(len(data)):
        predict = NN.predict(X[i])
        prob = sigmoid(predict)
        val = 0 if prob < 0.5 else 1
        if y[i] != val:
            times_wrong += 1

    return times_wrong / len(data)


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


if __name__ == '__main__':
    csv_train = "../DataSets/bank-note/train.csv"
    csv_test = "../DataSets/bank-note/test.csv"
    csv_testing = '../DataSets/test/testNN.csv'

    train_data = read_csv(csv_train)
    test_data = read_csv(csv_test)
    testing = read_csv(csv_testing)

    widths = [5, 10, 25, 50, 100]
    for w in widths:
        NN = NeuralNet(w, train_data)
        NN.stoch_grad_descent(0.1, 0.01, 10)
        print("Width: ", w)
        print("Training Error: ", NN_error(NN, train_data))
        print("Test Error: ", NN_error(NN, test_data))
        print()

    # w_0 = np.array([[0, 0, 0], [-1, -2, -3], [1, 2, 3]])
    # w_1 = np.array([[0, 0, 0], [-1 , -2, -3], [1, 2, 3]])
    # w_2 = np.array([[-1, 2, -1.5]])
    # test_weights = [w_0, w_1, w_2]
    # test_weights = np.array(test_weights)
    # test_sample = np.array([1, 1, 1])
    # NN = NeuralNet(3, testing, weights=test_weights)
    # NN.stoch_grad_descent(1, 1, 50)
    #
    # print()
    # for two_d in grad:
    #     print(two_d)
    #
    # print()
    # for two_d in NN.weights:
    #     print(two_d)

