import math
import numpy as np
import uuid
#from graphviz import Digraph


def process_numeric_data(data, numeric_atts):
    # Find median value for each numeric attribute
    for attribute in numeric_atts:
        numeric_vals_list = numeric_atts[attribute]
        numeric_atts[attribute] = np.median(np.array(numeric_vals_list))

    # Turns values from continuous data to binary in data based on median
    for sample in data:
        for attribute in numeric_atts:
            sample_val = int(sample[attribute])
            att_median = numeric_atts[attribute]

            sample[attribute] = -1 if sample_val <= att_median else 1

    return data


def fill_unknown_values(data, atts_with_unknown_val):
    most_common_val_of_att = dict()

    for att in atts_with_unknown_val:
        att_count = count_values_of_attribute_in_data(data, att)
        del att_count['unknown']
        most_common_val_of_att[att] = max(att_count, key=att_count.get)

    for sample in data:
        for att in atts_with_unknown_val:
            if sample[att] == 'unknown':
                sample[att] = most_common_val_of_att[att]

    return data


def read_csv(CSVfile, ordered_atts, numeric_atts, atts_with_unknown_val):

    with open(CSVfile, 'r') as f:
        data = list()

        for line in f:
            sample = line.strip().split(',')
            sample_to_add = dict()

            for x in range(len(sample) - 1):
                attribute = ordered_atts[x]
                value = sample[x]

                if attribute in numeric_atts:
                    numeric_atts[attribute].append(int(value))

                sample_to_add[attribute] = value

            sample_to_add['label'] = sample[len(sample) - 1]
            data.append(sample_to_add)

        if atts_with_unknown_val is not None:
            # Fills unknown values with its attribute's most common value
            data = fill_unknown_values(data, atts_with_unknown_val)

        if numeric_atts is not None:
            # Converts continuous data to binary based on attribute's median in set
            data = process_numeric_data(data, numeric_atts)

        return data


def read_txt_set_attr(TXTfile, fill_unknown):
    with open(TXTfile, 'r') as f:
        attributes = dict()
        ordered_atts = list()
        numeric_atts = dict()
        atts_with_unknown_val = set()

        for line in f:
            att_vals = line.split()
            att_name = att_vals[0]
            ordered_atts.append(att_name)
            attributes[att_name] = set()

            # Turn attribute's continuous values to binary
            if att_vals[1] == 'numeric':
                attributes[att_name].add(1)
                attributes[att_name].add(-1)
                numeric_atts[att_name] = list()
            else:
                for x in range(1, len(att_vals)):
                    value = att_vals[x]

                    if fill_unknown and value == 'unknown':
                        atts_with_unknown_val.add(att_name)
                        continue

                    attributes[att_name].add(value)

        return attributes, ordered_atts, numeric_atts, atts_with_unknown_val


def make_d_tree(data, attributes, **kwargs):
    information_gain_method = kwargs.get('gain', calc_entropy)
    max_depth = kwargs.get('max_depth', np.inf)
    should_bag = kwargs.get('should_bag', False)
    weights_of_samples_list = kwargs.get('weights', None)

    return id3(data, attributes, 0, max_depth, information_gain_method, weights_of_samples_list, should_bag)


def id3(data, attributes, depth, max_depth, information_gain_method, weights_of_samples_list, should_bag):

    label_pdf = count_values_of_attribute_in_data(data, 'label', weights_of_samples_list)

    node = dict()

    if len(label_pdf.keys()) == 1:
        node['label'] = next(iter(label_pdf.keys()))
        return node

    if len(attributes.keys()) == 0 or depth == max_depth:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    total_gain = information_gain_method(label_pdf, len(data))

    best_attr_key = find_best_att(data, attributes, total_gain, information_gain_method, weights_of_samples_list)

    node['attribute'] = best_attr_key
    best_attr_vals = attributes[best_attr_key]
    attributes_copy = attributes.copy()
    del attributes_copy[best_attr_key]
    node['values'] = dict()

    for value in best_attr_vals:
        data_subset = [sample for sample in data if sample[best_attr_key] == value]

        if len(data_subset) == 0:
            node['values'][value] = {'label': max(label_pdf, key=label_pdf.get)}
        else:
            node['values'][value] = id3(data_subset, attributes_copy, depth + 1, max_depth, information_gain_method, weights_of_samples_list, should_bag)

    return node


def find_best_att(data, attributes, tot_gain, information_gain_method, weights_of_samples_list):
    best_att = None
    max_gain = -1
    for att in attributes.keys():
        gain = tot_gain
        for value in attributes[att]:
            data_subset = [sample for sample in data if sample[att] == value]
            label_pdf = count_values_of_attribute_in_data(data_subset, 'label', weights_of_samples_list)

            weight = len(data_subset) / len(data)
            entropy = information_gain_method(label_pdf, len(data_subset))
            gain = gain - weight * entropy

        if gain >= max_gain:
            max_gain = gain
            best_att = att

    return best_att


def calc_entropy(labels_pdf, set_size):
    entropy = 0
    for label in labels_pdf:
        count = labels_pdf[label]
        prob = count / set_size
        entropy += -prob * math.log(prob, 2)

    return entropy


def calc_gini(labels_pdf, set_size):
    gini = 1
    for label in labels_pdf:
        count = labels_pdf[label]
        prob = count / set_size
        gini += -(prob ** 2)

    return gini


def calc_majority_error(labels_pdf, set_size):
    if set_size == 0:
        return 0

    max = 0
    for label in labels_pdf:
        if labels_pdf[label] > max:
            max = labels_pdf[label]

    return (set_size - max) / set_size


def count_values_of_attribute_in_data(data, attribute, weights_of_samples_list):

    value_counter = dict()

    for x in range(len(data)):
        sample = data[x]
        value = sample[attribute]
        if value not in value_counter:
            value_counter[value] = 0

        if weights_of_samples_list is None:
            value_counter[value] += 1
        else:
            value_counter[value] += weights_of_samples_list[x] #CHANGED

    return value_counter


def draw_tree(root, filename):
    graph = Digraph('Pretty-Decision-Tree', filename)
    visit(None, root, graph)
    graph.view()


def visit(parent, node, graph, key=None):
    id = uuid.uuid4().hex
    if isinstance(node, dict):
        if 'attribute' in node:
            graph.node(id, node['attribute'])
        else:
            graph.node(id, node['label'])
            graph.edge(parent, id, label=key)
            return
        if parent is not None:
            graph.edge(parent, id, label=key)
        for key, value in node['values'].items():
            visit(id, value, graph, key)


def test_tree_accuracy(test_data, root):

    if len(test_data) == 0:
        return 1

    num_correct_prediction = 0

    for sample in test_data:
        predicted_label = get_predicted_label_from_tree(root, sample)

        if predicted_label == sample['label']:
            num_correct_prediction += 1

    return num_correct_prediction / len(test_data)


def get_tree_weight_error_and_flag_correctly_predicted_samples(root, data, weights_of_samples):

    sum = 0
    sample_was_predicted_correctly = [0] * len(data)

    for x in range(len(data)):
        sample = data[x]
        predicted_label = get_predicted_label_from_tree(root, sample)

        if predicted_label == sample:
            sum += weights_of_samples[x]
            sample_was_predicted_correctly[x] = 1
        else:
            sum -= weights_of_samples[x]
            sample_was_predicted_correctly[x] = -1

        sum += weights_of_samples[x] if predicted_label == sample else -weights_of_samples[x]

    return 0.5 - 0.5 * sum, sample_was_predicted_correctly


def get_predicted_label_from_tree(root, sample):

    curr = root

    while 'label' not in curr:
        curr_att = curr['attribute']
        sample_att_val = sample[curr_att]

        curr = curr['values'][sample_att_val]

    return curr['label']


def find_average_accuracy_different_max_depths(training_data, test_data, attributes, num_trials, max_depth):
    gain_methods = {"Entropy": calc_entropy, "Majority Error": calc_majority_error, "Gini-Index": calc_gini}

    for x in range(1, max_depth + 1):
        print("depth " + str(x))

        train_acc_sum = {"Entropy": 0, "Majority Error": 0, "Gini-Index": 0}
        test_acc_sum = {"Entropy": 0, "Majority Error": 0, "Gini-Index": 0}
        for gain_method in gain_methods:
            for z in range(num_trials):
                att_copy = attributes.copy()
                root = id3(training_data, att_copy, 0, x, gain_methods[gain_method])
                train_acc_sum[gain_method] += test_tree_accuracy(training_data, root)
                test_acc_sum[gain_method] += test_tree_accuracy(test_data, root)

        for gain_method in gain_methods:
            print(gain_method)
            print("Training: ", train_acc_sum[gain_method] / num_trials, "\t test: ", test_acc_sum[gain_method] / num_trials)

        print()


def get_atts_and_test_and_training_data_from_file(attributes_file, training_data_file, test_data_file, **kwargs):
    fill_unknown = kwargs.get("fill_unknown", False)

    attributes, ordered_atts, numeric_atts, atts_with_unknown_val = read_txt_set_attr(attributes_file, fill_unknown)
    numeric_atts_copy = numeric_atts.copy()
    training_data = read_csv(training_data_file, ordered_atts, numeric_atts, atts_with_unknown_val)
    test_data = read_csv(test_data_file, ordered_atts, numeric_atts_copy, atts_with_unknown_val)

    return attributes, training_data, test_data

def main():
    attributes, training_data, test_data = get_atts_and_test_and_training_data_from_file("car/data-desc-readable.txt", "car/train.csv", "car/test.csv")
    root = make_d_tree(training_data, attributes, gain=calc_majority_error)
    print(root)
    #find_average_accuracy_different_max_depths(training_data, test_data, attributes, 10, 6)



    #test
    # attributes, ordered_atts = read_txt_set_attr("TestTennis/playtennislabels.txt")
    # training_data = read_csv("TestTennis/playtennis.csv", ordered_atts)
    # test_data = [{'Outlook:': 'Sunny', 'Temperature:': 'Hot', 'Humidity:': 'High', 'Wind:': 'Strong', 'label': 'Yes'}]
    # find_accuracy_different_max_depths(training_data, test_data, attributes, 4)


if __name__ == "__main__": main()