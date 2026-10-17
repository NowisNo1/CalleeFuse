import json

from torch.utils.data import Dataset


def read_gson(path):
    # Read the file
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_gsons(paths):
    data = []
    for path in paths:
        data.extend(read_gson(path))
    return data


class CalibrationDataset(Dataset):
    def __init__(self, data, tokenizer, parser, max_length=512, label_length=16):
        self.data = data
        self.tokenizer = tokenizer
        self.parser = parser
        self.max_length = max_length
        self.label_length = label_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.parser([self.data[idx]], self.tokenizer, self.max_length, self.label_length)


class CallTreeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_node = 0
        self.get_max_node()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

    def get_max_node(self):
        node_sizes = []
        for i in range(len(self.data)):
            item = self.data[i]
            # Example: process text and labels (adjust to your data fields)
            root = item["root"]
            method_full_name = root["method_full_name"]
            method_body = root["method_body"]
            masked_root_method = method_body

            nodes, edges = self.parse(root, 0, [masked_root_method], [])
            node_sizes.append(len(nodes))
            self.max_node = max(self.max_node, len(nodes))

    def parse(self, root, current_idx, nodes, edges):
        if "children" not in root.keys():
            return nodes, edges
        children = root["children"]
        if len(children) > 0:
            for i in range(len(children)):
                method = children[i]["method_body"]
                if method == "unResolve":
                    continue
                nodes.append(method.strip())
                edges.append([current_idx, current_idx + i + 1])
                self.parse(children[i], current_idx + i + 1 + len(children), nodes, edges)

        return nodes, edges



