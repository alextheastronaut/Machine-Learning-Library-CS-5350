import numpy as np
import matplotlib.pyplot as plt
from DecisionTree import make_d_tree, get_tree_weight_error_and_flag_correctly_predicted_samples, \
    get_atts_and_test_and_training_data_from_file, get_predicted_label_from_tree


def create_ada_boosted_stumps(data, attributes, num_stumps):
    stumps = [None] * num_stumps
    weights_of_samples = np.array([1 / len(data) for x in range(len(data))])

    for x in range(num_stumps):
        print(x)
        root = make_d_tree(data, attributes, max_depth=1, weights=weights_of_samples)
        tree_weighted_error, was_correctly_predicted = get_tree_weight_error_and_flag_correctly_predicted_samples(root,
                                                                                                                  data,
                                                                                                                  weights_of_samples)
        tree_say = np.log((1 - tree_weighted_error) / tree_weighted_error)
        update_and_normalize_weights(weights_of_samples, was_correctly_predicted, tree_say)
        #print("Error ", x, "\t", tree_weighted_error)
        #print("Alpha ", tree_say)
        #print(root)

        stumps[x] = dict()
        stumps[x]["stump"] = root
        stumps[x]["say"] = tree_say

    return stumps


def error_vs_num_trees(num_iter, attributes, training_data, test_data):
    test_error_at_t = [0] * num_iter
    train_error_at_t = [0] * num_iter

    #stumps
    trees = create_ada_boosted_stumps(training_data, attributes, num_iter)
    #bagged
    #trees = bagged_trees_or_rand_forest(training_data, attributes, num_iter, -1)

    for num_trees in range(1, num_iter + 1):
        print(num_trees)
        #stumps
        train_error_at_t[num_trees - 1] = find_error_of_boosted_stumps(trees, num_trees, training_data, True)
        test_error_at_t[num_trees - 1] = find_error_of_boosted_stumps(trees, num_trees, test_data, True)

        #bagged
        #trees = bagged_trees_or_rand_forest(training_data, attributes, num_trees, -1)
        # train_error_at_t[num_trees - 1] = find_error_of_boosted_stumps(trees, num_trees, training_data, False)
        # test_error_at_t[num_trees - 1] = find_error_of_boosted_stumps(trees, num_trees, test_data, False)

    x = [i for i in range(1, num_iter + 1)]
    #stumps
    make_figure(x, train_error_at_t, test_error_at_t, "Ada Boosted Stumps", "Number of Stumps", "Error", "HW2stumps")
    #bagged
    #make_figure(x, train_error_at_t, test_error_at_t, "Bagged Decision Trees", "Number of Trees", "Error", "HW2test")


def make_figure(x, y1, y2, graph_title, x_label, y_label, picture_name):
    fig = plt.figure()
    plt.plot(x, y1, color="blue", label='Training')
    plt.plot(x, y2, color="red", label='Test')
    plt.title(graph_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    path = "/home/alex/MachineLearning/pics/"
    fig.savefig(path + picture_name + '.png', dpi=100)
    plt.show()


def update_and_normalize_weights(weights_of_samples, was_correctly_predicted, tree_say):
    for x in range(len(weights_of_samples)):
        weights_of_samples[x] *= np.exp(-tree_say * was_correctly_predicted[x])

    hi = np.sum(weights_of_samples)
    weights_of_samples /= np.sum(weights_of_samples)


def find_error_of_boosted_stumps(trees, num_trees, test_data, are_stumps):
    sum_correct = 0
    for sample in test_data:
        sum_vote = 0

        for x in range(num_trees):

            if are_stumps:
                predicted_label = get_predicted_label_from_tree(trees[x]['stump'], sample)
                sum_vote += trees[x]['say'] if predicted_label == 'yes' else -trees[x]['say']
            else:
                predicted_label = get_predicted_label_from_tree(trees[x], sample)
                sum_vote += 1 if predicted_label == 'yes' else -1

        vote_predicted_label = 'yes' if np.sign(sum_vote) >= 0 else 'no'

        if vote_predicted_label == sample['label']:
            sum_correct += 1

    return (len(test_data) - sum_correct) / len(test_data)


def bagged_trees_or_rand_forest(data, attributes, num_trees, att_subset_size):
    trees = [None] * num_trees

    for x in range(num_trees):
        print(x)
        bag = [None] * len(data)

        for y in range(len(data)):
            bag[y] = data[np.random.randint(0, len(data))]

        trees[x] = make_d_tree(bag, attributes, att_subset_size = att_subset_size)

    return trees


if __name__ == "__main__":
    attributes, training_data, test_data = get_atts_and_test_and_training_data_from_file(
        "../DataSets/bank/data-desc-readable.txt", "../DataSets/bank/train.csv", "../DataSets/bank/test.csv")
    #stumps = create_ada_boosted_stumps(training_data, attributes, 100)
    #print(find_error_of_boosted_stumps(stumps, test_data, True))
    #trees = bagged_trees_or_rand_forest(training_data, attributes, 20, True, 4)
    #print(find_error_of_boosted_stumps(stumps, training_data, True))
    error_vs_num_trees(1000, attributes, training_data, test_data)
