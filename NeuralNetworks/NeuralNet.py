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

        self.weights = kwargs.get('weights', np.zeros((3, d, d)))
        self.cache = np.zeros((3, d))
        b_nodes = np.array([[1] * 3])
        self.nodes = np.concatenate((b_nodes.transpose(), np.zeros((3, d - 1))), axis=1)

    def predict(self, x_i):
        self.nodes[0] = x_i
        self.update_sigmoid_layer(1, len(x_i))
        self.update_sigmoid_layer(2, self.width)

        return self.weights[2][0].dot(self.nodes[2])

    def update_sigmoid_layer(self, layer_idx, num_nodes):
        for node_idx in range(1, num_nodes):
            print(self.weights[layer_idx - 1][node_idx])
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
                self.weights = self.weights - r * grad

    def back_prop(self, x_i, y):
        cache = np.zeros(self.nodes.shape)
        grad = np.zeros(self.weights.shape)

        # Cache y and calc gradient in last hidden layer
        cache[2][0] = self.predict(x_i) - y
        for n in range(self.width):
            grad[2][0][n] = cache[2][0] * self.nodes[2][n]

        # back propogate 2nd then 1st hidden layer
        self.back_prop_layer(2, self.width, self.width, cache, grad)
        self.back_prop_layer(1, len(x_i), self.width, cache, grad)

        print(cache)
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
            val = self.nodes[layer][node_1] * (1 - self.nodes[layer][node_1])

            #Populate cache for layer

            # Vector of outgoing edges from node_1
            outgoing_edges = self.weights[layer][:, node_1]
            delta = cache[layer].dot(outgoing_edges)
            cache[layer - 1][node_1] = delta * val

            # For each ingoing edge to node_1
            for edge in range(num_nodes_0):
                grad[layer - 1][node_1][edge] = delta * val * self.nodes[layer - 1][edge]


                #cache_res = 0
                #for node_1 in range(1, num_nodes_1):
                #    cache_res += cache_res[layer_idx + 1][node_1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def learn_rate(gamma, d, t):
    return gamma / (1 + gamma * t / d)


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


if __name__ == '__main__':
    csv_train = "../DataSets/bank-note/train.csv"
    csv_test = "../DataSets/bank-note/test.csv"
    csv_testing = '../DataSets/test/testNN.csv'

    train_data = read_csv(csv_train)
    test_data = read_csv(csv_test)
    testing = read_csv(csv_testing)

    w_0 = np.array([[0, 0, 0], [-1, -2, -3], [1, 2, 3]])
    w_1 = np.array([[0, 0, 0], [-1 , -2, -3], [1, 2, 3]])
    w_2 = np.array([[-1, 2, -1.5], [0, 0, 0], [0, 0, 0]])
    test_weights = [w_0, w_1, w_2]
    test_weights = np.array(test_weights)
    test_sample = np.array([1, 1, 1])
    NN = NeuralNet(3, testing, weights=test_weights)
    grad = NN.back_prop(test_sample, 1)

    print()
    for two_d in grad:
        print(two_d)

    print()
    for two_d in NN.weights:
        print(two_d)

