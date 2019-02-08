def read_csv(CSVfile):
    with open(CSVfile, 'r') as f:
        for line in f:
            terms = line.strip().split(',')


def read_and_set_attr_txt(TXTfile):
    with open(TXTfile, 'r') as f:
        self.labels = f.readline()
        for line in f:
            att_vals = line.split()
            att_name = line[0]
            self.attributes[att_name] = set()
            self.attr_ordered.append(att_name)
            for x in range(1, len(att_vals)):
                self.attributes[att_name].add(att_vals[x])


def id3(data, attributes, information_gain_method):
    label_pdf = count_labels_in_data(data)

    node = dict()

    if len(label_pdf.keys()) == 1:
        node['label'] = next(iter(label_pdf.keys()))
        return node

    if len(attributes.keys()) == 0:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    total_gain = information_gain_method(data)

    #Finds best attribute key, returns none if max info gain is 0
    best_attr_key = find_best_att(data, attributes, total_gain, information_gain_method)

    if best_attr_key is None:
        node['label'] = max(label_pdf, key=label_pdf.get)
        return node

    node['attribute'] = best_attr_key
    best_attr_vals = attributes[best_attr_key]
    del attributes[best_attr_key]
    for value in best_attr_vals:
        node['children'] = dict()
        data_subset = [sample for sample in data if sample[best_attr_key] == value]

        if (len(data_subset) == 0):
            node['children'][value] = {'label': max(label_pdf, key=label_pdf.get)}
        else:
            node['children'][value] = id3(data, attributes, information_gain_method)

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
            gain = gain - weight * information_gain_method(label_pdf)

        if gain > max_gain:
            max_gain = gain
            best_att = att

    return best_att

def count_labels_in_data(data):
    label_counter = dict()
    for sample in data:
        label = sample['label']
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1

    return label_counter


def main():
    read_csv("car/train.csv")


if __name__ == "__main__": main()
