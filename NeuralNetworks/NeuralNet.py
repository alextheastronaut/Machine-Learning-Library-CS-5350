import numpy as np

class NN:
    def __init__(self, width, training_data):
        self.width = width

        # Separate attributes from labels
        n, d = training_data.shape
        self.y = training_data[:, d]
        self.X = training_data[:, : d]
        b = np.array([[1] * n])
        # Prepend bias column to training matrix
        self.X = np.concatenate((b.transpose(), self.X), axis=1)

        self.weights = np.zeros((3, d, d - 1))
        self.nodes = np.zeroes((3, d))


    def predict(self):
        print()


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

    train_data = read_csv(csv_train)
    test_data = read_csv(csv_test) 