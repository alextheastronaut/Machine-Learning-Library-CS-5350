import numpy as np
from DecisionTree.DecisionTree import make_d_tree, get_tree_weight_error_and_flag_correctly_predicted_samples, \
    get_atts_and_test_and_training_data_from_file, get_predicted_label_from_tree


def create_ada_boosted_stumps(data, attributes, num_stumps):
    stumps = [None] * num_stumps
    weights_of_samples = np.array([1 / len(data) for x in range(len(data))])

    for x in range(num_stumps):
        root = make_d_tree(data, attributes, max_depth=1, weights=weights_of_samples)
        tree_weighted_error, was_correctly_predicted = get_tree_weight_error_and_flag_correctly_predicted_samples(root,
                                                                                                                  data,
                                                                                                                  weights_of_samples)
        tree_say = np.log((1 - tree_weighted_error) / tree_weighted_error)
        update_and_normalize_weights(weights_of_samples, was_correctly_predicted, tree_say)
        #print("Error ", x, "\t", tree_weighted_error)

        stumps[x] = dict()
        stumps[x]["stump"] = root
        stumps[x]["say"] = tree_say

    return stumps


def update_and_normalize_weights(weights_of_samples, was_correctly_predicted, tree_say):
    for x in range(len(weights_of_samples)):
        weights_of_samples[x] *= np.exp(-tree_say * was_correctly_predicted[x])

    weights_of_samples /= np.sum(weights_of_samples)


def find_error_of_boosted_stumps(stumps, test_data):
    sum_correct = 0
    for sample in test_data:
        sum_vote = 0

        for stmp in stumps:
            predicted_label = get_predicted_label_from_tree(stmp['stump'], sample)
            sum_vote += stmp['say'] if predicted_label == 'yes' else -stmp['say']

        stump_predicted_label = 'yes' if np.sign(sum_vote) >= 0 else 'no'

        if stump_predicted_label == sample['label']:
            sum_correct += 1

    return sum_correct / len(test_data)


if __name__ == "__main__":
    attributes, training_data, test_data = get_atts_and_test_and_training_data_from_file(
        "../DataSets/bank/data-desc-readable.txt", "../DataSets/bank/train.csv", "../DataSets/bank/test.csv")
    stumps = create_ada_boosted_stumps(training_data, attributes, 1)
    print(find_error_of_boosted_stumps(stumps, test_data))
