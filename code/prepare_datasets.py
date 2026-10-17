import random
import re

import matplotlib
import matplotlib.pyplot as plt
import json
from transformers import AutoTokenizer

from gson_reader import read_gsons

model_path = "models/CodeT5p-220m"
# trained_model_path = "../saved_model/epoch_1"
# model_path = trained_model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
data_path = "models/model-2025-10-15/Depth2RandomMaskNoTest.json"
matplotlib.use("TKAgg")


def parse(root, current_idx, nodes, edges, depth, threshold, num_son):
    if "children" not in root.keys() or depth == threshold:
        return nodes, edges, num_son
    children = root["children"]
    if len(children) > 0:
        for i in range(len(children)):
            method = children[i]["method_body"]
            tokens = tokenizer.tokenize(method)
            if len(tokens) > 128:
                return [], [], -1

            num_son += 1
            nodes.append([method.strip(), num_son])
            edges.append([current_idx, num_son])
            _, _, num_son = parse(children[i], num_son, nodes, edges, depth + 1, threshold, num_son)
            if num_son == -1:
                return [], [], -1
    return nodes, edges, num_son


def camel_case_tokenize(text):
    return re.sub('([a-z])([A-Z])', r'\1 \2', text).split()


# new_data = []
# cnt = 0
# with open(data_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
#     print(len(data))
#     for jobj in data:
#         root = jobj["root"]
#         method_body = root["method_body"]
#         masked_root_method = method_body
#         nodes, edges, num_son = parse(root, 0, [[method_body, 0]], [], 0, 1, 0)
#         if num_son == -1:
#             continue
#         method_full_name = root["method_full_name"]
#         pos = method_full_name.rfind('.')
#         pos_end = method_full_name.rfind("(")
#         label = method_full_name
#         if pos != -1 and pos_end != -1:
#             label = method_full_name[pos + 1: pos_end]
#         if "_" in label or '$' in label:
#             continue
#         tokens = camel_case_tokenize(label)
#
#         if len(tokenizer.tokenize(label)) > 16:
#             continue
#         if num_son > 20:
#             continue
#         f = False
#         for i in range(len(tokens)):
#             if tokens[i] in tokens[i + 1:]:
#                 f = True
#                 break
#         if f:
#             continue
#         if len(tokenizer.tokenize(masked_root_method)) > 128:
#             continue
#         cnt += 1
#         new_data.append(jobj)
#     print(1.0 * cnt / len(data))
# # # # plt.hist(new_data, bins="auto")
# # # # plt.show()
# with open("d_2_b_128_l_16_random_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=2)

# d_1 = read_gsons(["d_1_b_128_l_16_no_test.json"])
# d_2_mask = read_gsons(["d_2_b_128_l_16_mask_no_test.json"])
# d_2_random = read_gsons(["d_2_b_128_l_16_random_no_test.json"])
# d_1_new = []
# d_2_mask_new = []
# d_2_random_new = []
#
# d_1_maps = {}
# d_2_mask_maps = {}
# d_2_random_maps = {}
#
# for obj in d_1:
#     if obj["root"]["hashCode"] in d_1_maps.keys():
#         print(obj["root"]["hashCode"])
#     else:
#         d_1_maps[obj["root"]["hashCode"]] = obj
#
# for obj in d_2_mask:
#     if obj["root"]["hashCode"] in d_2_mask_maps.keys():
#         print(obj["root"]["hashCode"])
#     else:
#         d_2_mask_maps[obj["root"]["hashCode"]] = obj
#
# for obj in d_2_random:
#     if obj["root"]["hashCode"] in d_2_random_maps.keys():
#         print(obj["root"]["hashCode"])
#     else:
#         d_2_random_maps[obj["root"]["hashCode"]] = obj
#
# filter_keys = set(d_1_maps.keys()) & set(d_2_mask_maps.keys()) & set(d_2_random_maps.keys())
#
# for hash_code in filter_keys:
#     d_1_new.append(d_1_maps[hash_code])
#     d_2_mask_new.append(d_2_mask_maps[hash_code])
#     d_2_random_new.append(d_2_random_maps[hash_code])
#
# with open("filter_d_1_b_128_l_16_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_1_new, f, ensure_ascii=False, indent=2)
#
# with open("filter_d_2_b_128_l_16_mask_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_mask_new, f, ensure_ascii=False, indent=2)
#
# with open("filter_d_2_b_128_l_16_random_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_random_new, f, ensure_ascii=False, indent=2)


# d_1_train = read_gsons(["train_d_1_b_128_l_16_no_test.json"])
# d_1_test = read_gsons(["test_d_1_b_128_l_16_no_test.json"])
# d_2_mask = read_gsons(["d_2_b_128_l_16_mask_no_test.json"])
# d_2_random = read_gsons(["d_2_b_128_l_16_random_no_test.json"])
# # cnt = 0
# train_map = {}
# test_map = {}
# d_2_mask_map = {}
# d_2_random_map = {}
# d_2_mask_train = []
# d_2_mask_test = []
# d_2_random_train = []
# d_2_random_test = []
# for obj in d_1_train:
#     train_map[obj["root"]["hashCode"]] = obj
#
# for obj in d_1_test:
#     test_map[obj["root"]["hashCode"]] = obj
#
# for obj in d_2_mask:
#     d_2_mask_map[obj["root"]["hashCode"]] = obj
#
# for obj in d_2_random:
#     d_2_random_map[obj["root"]["hashCode"]] = obj
#
#
# for hash_code in train_map.keys():
#     d_2_random_train.append(d_2_random_map[hash_code])
#     d_2_mask_train.append(d_2_mask_map[hash_code])
#
# for hash_code in test_map.keys():
#     d_2_random_test.append(d_2_random_map[hash_code])
#     d_2_mask_test.append(d_2_mask_map[hash_code])
#
# with open("train_d_2_b_128_l_16_mask_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_mask_train, f, ensure_ascii=False, indent=2)
#
# with open("test_d_2_b_128_l_16_mask_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_mask_test, f, ensure_ascii=False, indent=2)
#
# with open("train_d_2_b_128_l_16_random_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_random_train, f, ensure_ascii=False, indent=2)
#
# with open("test_d_2_b_128_l_16_random_no_test.json", "w", encoding="utf-8") as f:
#     json.dump(d_2_random_test, f, ensure_ascii=False, indent=2)


mask_recursive = read_gsons(["models/model-2025-10-15/Depth3MaskRecursiveSub.json"])
old = read_gsons(["models/model-2025-10-15/test_d_2_b_128_l_16_mask_no_test.json"])

old_map = {}
mask_recursive_map = {}

for obj in old:
    if obj["root"]["hashCode"] in old_map.keys():
        print(obj["root"]["hashCode"])
    else:
        old_map[obj["root"]["hashCode"]] = obj

for obj in mask_recursive:
    if obj["root"]["hashCode"] in mask_recursive_map.keys():
        print(obj["root"]["hashCode"])
    else:
        mask_recursive_map[obj["root"]["hashCode"]] = obj

new_data = []

for obj in old:
    new_data.append(mask_recursive_map[obj["root"]["hashCode"]])

with open("test_d_2_b_128_l_16_mask_recursive_no_test.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)




