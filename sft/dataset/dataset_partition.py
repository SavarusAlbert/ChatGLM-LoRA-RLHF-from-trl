import json
import random

train_data = []
for line in open("train.json", "r", encoding="utf-8"):
    train_data.append(json.loads(line))

train_data_dict = dict()
for data in train_data:
    if data['task_dataset'] not in train_data_dict:
        train_data_dict[data['task_dataset']] = []
    train_data_dict[data['task_dataset']].append(data)

partition = 1 / 10
new_train_data, rlhf_data = [], []
for key in train_data_dict:
    value = train_data_dict[key]
    value_len = len(value)
    n_slice = round(value_len * partition)
    new_train_data += value[n_slice:]
    rlhf_data += value[:n_slice]

# 打乱数据
random.shuffle(new_train_data)
random.shuffle(rlhf_data)

# 写入数据
with open("new_train.json", "w", encoding="utf-8") as fn:
    for data in new_train_data:
        json.dump(data, fn, ensure_ascii=False)
        fn.write("\n")

with open("rlhf.json", "w", encoding="utf-8") as fn:
    for data in rlhf_data:
        json.dump(data, fn, ensure_ascii=False)
        fn.write("\n")