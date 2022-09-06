import os


def generate_fewshot_data(ori_train_path, target_path, num_each_cls=5):
    # 均衡类别采样
    ori_train_in = readfile(os.path.join(ori_train_path, "train/intent_seq.in"))
    ori_train_out = readfile(os.path.join(ori_train_path, "train/intent_seq.out"))
    slot_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_train_path, "vocab/slot_vocab"))]
    dic = {x: [] for x in slot_label_list}  # slot: [[i,o]]
    for i, o in zip(ori_train_in, ori_train_out):
        x1, x2, x3 = statistic(i, o, dic)

    fewshot_train_in = []
    fewshot_train_out = []
    for k, v in dic.items():
        in_list = [x[0] for x in v]
        out_list = [x[1] for x in v]

        fewshot_train_in.extend(in_list[:num_each_cls])
        fewshot_train_out.extend(out_list[:num_each_cls])

    target_path = os.path.join(target_path, "fewshot-" + str(num_each_cls))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    writefile(fewshot_train_in, os.path.join(target_path, "intent_seq.in"))
    writefile(fewshot_train_out, os.path.join(target_path, "intent_seq.out"))

def generate_fewshot_data_mj(ori_train_path, target_path, num_each_cls, label_to_num):
    # 模拟工业界采样过程
    ori_train_in = readfile(os.path.join(ori_train_path, "train/intent_seq.in"))
    ori_train_out = readfile(os.path.join(ori_train_path, "train/intent_seq.out"))
    slot_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_train_path, "vocab/slot_vocab"))]
    dic = {x: [] for x in slot_label_list}  # slot: [[i,o]]
    for i, o in zip(ori_train_in, ori_train_out):
        x1, x2, x3 = statistic(i, o, dic)

    fewshot_train_in = []
    fewshot_train_out = []
    for k, v in dic.items():
        in_list = [x[0] for x in v]
        out_list = [x[1] for x in v]

        fewshot_train_in.extend(in_list[:label_to_num[k]])
        fewshot_train_out.extend(out_list[:label_to_num[k]])

    target_path = os.path.join(target_path, "fewshot-" + str(num_each_cls))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    writefile(fewshot_train_in, os.path.join(target_path, "intent_seq.in"))
    writefile(fewshot_train_out, os.path.join(target_path, "intent_seq.out"))
    return target_path

def statistic(i, o, dic):
    followed_by_entity = 0  # 实体边界后接上另一个实体词
    followed_by_nonent = 0  # 实体边界后接上非实体词
    num_same_neighbor = 0  # 连续相同类型实体　B-time B-time

    intent = o.split()[0]
    slots = o.split()[1:]
    slots.append('O')  # 添加虚拟尾节点
    stack = []
    idx = 0
    while idx < len(slots):
        if slots[idx] == 'O':
            if stack:
                followed_by_nonent += 1
                single_slot_label = [intent] + generate_single_slot_label(slots[:-1], idx - len(stack), idx - 1)
                dic[stack[-1].split('-')[1]].append([i, " ".join(single_slot_label)])
                stack = []
            idx += 1
        else:
            if stack:
                pre_slot = stack[-1]
                if pre_slot.split('-')[1] == slots[idx].split('-')[1] and slots[idx].split('-')[0] != 'B':
                    stack.append(slots[idx])
                    idx += 1
                else:
                    if pre_slot.split('-')[1] == slots[idx].split('-')[1] and slots[idx].split('-')[0] == 'B':
                        num_same_neighbor += 1
                    followed_by_entity += 1
                    single_slot_label = [intent] + generate_single_slot_label(slots[:-1], idx - len(stack), idx - 1)
                    dic[stack[-1].split('-')[1]].append([i, " ".join(single_slot_label)])
                    stack = []
            else:
                stack.append(slots[idx])
                idx += 1
    return followed_by_entity, followed_by_nonent, num_same_neighbor


def generate_single_slot_label(slots, start, end):
    new_slot = []
    for idx, slot in enumerate(slots):
        if idx < start or idx > end:
            if slot == 'O':
                new_slot.append(slot)
            else:
                new_slot.append('[PAD]')
        else:
            new_slot.append(slot)
    return new_slot


def readfile(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        res = [x.strip() for x in f.readlines()]
    return res


def writefile(str_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in str_list])


def average_sample(ori_path, target_path, num_cls):
    generate_fewshot_data(ori_path, target_path, num_each_cls=num_cls)

def mj_sample(ori_path, target_path, num_each_cls, label_to_num):
    data_path = generate_fewshot_data_mj(ori_path, target_path, num_each_cls, label_to_num)
    return data_path

