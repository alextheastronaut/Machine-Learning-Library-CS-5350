import math
import numpy as np
import uuid
from graphviz import Digraph

def process_numeric_data(data, numeric_atts):
    #Find median value for each numeric attribute
    for attribute in numeric_atts:
        numeric_vals_list = numeric_atts[attribute]
        numeric_atts[attribute] = np.median(np.array(numeric_vals_list))

    #Turns values from continuous data to binary in data based on median
    for sample in data:
        for attribute in numeric_atts:
            sample_val = int(sample[attribute])
            att_median = numeric_atts[attribute]

            sample[attribute] = '<=' if sample_val <= att_median else '>'

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


def read_csv(CSVfile, ordered_atts, numeric_atts, atts_with_unknown_val, fill_unknown):
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

        if fill_unknown:
            #Fills unknown values with its attribute's most common value
            data = fill_unknown_values(data, atts_with_unknown_val)

        #Converts continuous data to binary based on attribute's median in set
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

            #Turn attribute's continuous values to binary
            if att_vals[1] == 'numeric':
                attributes[att_name].add('>')
                attributes[att_name].add('<=')
                numeric_atts[att_name] = list()
            else:
                for x in range(1, len(att_vals)):
                    value = att_vals[x]

                    if fill_unknown and value == 'unknown':
                        atts_with_unknown_val.add(att_name)
                        continue

                    attributes[att_name].add(value)

        return attributes, ordered_atts, numeric_atts, atts_with_unknown_val


def id3(data, attributes, depth, max_depth, information_gain_method):
    label_pdf = count_values_of_attribute_in_data(data, 'label')

    node = dict()

    if len(label_pdf.keys()) == 1:
        node['label'] = next(iter(label_pdf.keys()))
        return node

    if len(attributes.keys()) == 0 or depth == max_depth:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    total_gain = information_gain_method(label_pdf, len(data))

    best_attr_key = find_best_att(data, attributes, total_gain, information_gain_method)

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
            node['values'][value] = id3(data_subset, attributes_copy, depth + 1, max_depth, information_gain_method)

    return node


def find_best_att(data, attributes, tot_gain, information_gain_method):
    best_att = None
    max_gain = -1
    for att in attributes.keys():
        gain = tot_gain
        for value in attributes[att]:
            data_subset = [sample for sample in data if sample[att] == value]
            label_pdf = count_values_of_attribute_in_data(data_subset, 'label')

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
        prob = count/set_size
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

    return (set_size - max)/set_size


def count_values_of_attribute_in_data(data, attribute):
    value_counter = dict()
    for sample in data:
        value = sample[attribute]
        if value not in value_counter:
            value_counter[value] = 0
        value_counter[value] += 1

    return value_counter


def draw_tree(root, filename):
    graph = Digraph('Pretty-Decision-Tree', filename)
    visit(None, root, graph)
    graph.view()

def visit(parent, node, graph, key = None):
    id = uuid.uuid4().hex
    if isinstance(node, dict):
        if 'attribute' in node:
            graph.node(id, node['attribute'])
        else:
            graph.node(id, node['label'])
            graph.edge(parent, id, label=key)
            return
        if parent != None:
            graph.edge(parent, id, label = key)
        for key, value in node['values'].items():
            visit(id, value, graph, key)

def test_tree_accuracy(test_data, root):
    if len(test_data) == 0:
        return 1

    num_correct_prediction = 0

    for sample in test_data:
        curr = root
        while 'label' not in curr:
            curr_att = curr['attribute']
            sample_att_val = sample[curr_att]

            curr = curr['values'][sample_att_val]

        predicted_label = curr['label']

        if predicted_label == sample['label']:
            num_correct_prediction += 1

    return num_correct_prediction / len(test_data)


def find_accuracy_different_max_depths(training_data, test_data, attributes, max_depth):

    gain_methods = [calc_entropy, calc_majority_error, calc_gini]

    for x in range(1, max_depth + 1):
        print("depth " + str(x))
        for gain_method in gain_methods:
            att_copy = attributes.copy()
            root = id3(training_data, att_copy, 0, x, gain_method)
            test_acc = test_tree_accuracy(test_data, root)
            train_acc = test_tree_accuracy(training_data, root)
            print("training: " + str(train_acc) + "\t" + "test: " + str(test_acc))

        print()


def main():
    attributes, ordered_atts, numeric_atts, atts_with_unknown_val = read_txt_set_attr("bank/data-desc-readable.txt", False)
    numeric_atts_copy = numeric_atts.copy()
    training_data = read_csv("bank/train.csv", ordered_atts, numeric_atts, atts_with_unknown_val, False)
    test_data = read_csv("bank/test.csv", ordered_atts, numeric_atts_copy, atts_with_unknown_val, False)
    #root = id3(training_data, attributes, 0, 4, calc_majority_error)
    find_accuracy_different_max_depths(training_data, test_data, attributes, 16)
    #draw_tree(root, "hi")
    #test
    #attributes, ordered_atts = read_txt_set_attr("TestTennis/playtennislabels.txt")
    #training_data = read_csv("TestTennis/playtennis.csv", ordered_atts)
    #test_data = [{'Outlook:': 'Sunny', 'Temperature:': 'Hot', 'Humidity:': 'High', 'Wind:': 'Strong', 'label': 'Yes'}]
    #find_accuracy_different_max_depths(training_data, test_data, attributes, 4)


if __name__ == "__main__": main()
