import math

def read_csv(CSVfile, ordered_atts):
    with open(CSVfile, 'r') as f:
        data = list()

        for line in f:
            sample = line.strip().split(',')
            sample_to_add = dict()

            for x in range(len(sample) - 1):
                attribute = ordered_atts[x]
                value = sample[x]
                sample_to_add[attribute] = value

            sample_to_add['label'] = sample[len(sample) - 1]
            data.append(sample_to_add)

        return data


def read_txt_set_attr(TXTfile):
    with open(TXTfile, 'r') as f:
        attributes = dict()
        ordered_atts = list()

        for line in f:
            att_vals = line.split()
            att_name = att_vals[0]
            ordered_atts.append(att_name)
            attributes[att_name] = set()

            for x in range(1, len(att_vals)):
                attributes[att_name].add(att_vals[x])

        return attributes, ordered_atts


def id3(data, attributes, depth, max_depth, information_gain_method):
    label_pdf = count_labels_in_data(data)

    node = dict()

    if len(label_pdf.keys()) == 1:
        node['label'] = next(iter(label_pdf.keys()))
        return node

    if len(attributes.keys()) == 0 or depth == max_depth:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    total_gain = information_gain_method(label_pdf, len(data))

    #Finds best attribute key, returns none if max info gain is 0
    best_attr_key = find_best_att(data, attributes, total_gain, information_gain_method)

    if best_attr_key is None:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    node['attribute'] = best_attr_key
    best_attr_vals = attributes[best_attr_key]
    del attributes[best_attr_key]
    node['children'] = dict()
    
    for value in best_attr_vals:
        data_subset = [sample for sample in data if sample[best_attr_key] == value]

        if (len(data_subset) == 0):
            node['children'][value] = {'label': max(label_pdf, key=label_pdf.get)}
        else:
            node['children'][value] = id3(data_subset, attributes, depth + 1, max_depth, information_gain_method)

    return node


def find_best_att(data, attribtues, tot_gain, information_gain_method):
    best_att = None
    max_gain = 0
    for att in attribtues.keys():
        gain = tot_gain
        for value in attribtues[att]:
            data_subset = [sample for sample in data if sample[att] == value]
            label_pdf = count_labels_in_data(data_subset)

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


def count_labels_in_data(data):
    label_counter = dict()
    for sample in data:
        label = sample['label']
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1

    return label_counter


def print_tree(root):
    q = list()
    q.append(root)
    while len(q) > 0:
        curr = q.pop(0)

        if 'children' in curr:
            for child in curr['children']:
                q.append(curr['children'][child])

        to_print = ''
        for key in curr:
            to_print += key + " " + str(curr[key]) + " "
        print(to_print + '\t')


def main():
    attributes, ordered_atts = read_txt_set_attr("TestTennis/playtennislabels.txt")
    data = read_csv("TestTennis/playtennis.csv", ordered_atts)
    root = id3(data, attributes, 0, 50, calc_majority_error)
    print_tree(root)

if __name__ == "__main__": main()
