import argparse
import os
from collections import Counter
import numpy as np
import json
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import NluDataset, glue_processor, prepare_data, construct_label_inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_raw, id_to_label, tokenizer, mode='dev'):
    slot_label_list = id_to_label['slot_labels']
    intent_label_list = id_to_label['intent_labels']
    slot_prompts = id_to_label['slot_prompts']
    model.eval()
    test_data = NluDataset(data_raw)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)

    joint_all = 0
    joint_correct = 0
    s_preds = []
    s_labels = []
    i_preds = []
    i_labels = []

    predicted_masked_tokens = []
    epoch_pbar = tqdm(test_dataloader, desc="Evaluation", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, slot_ids= batch
        with torch.no_grad():
            slot_logits = model(input_ids, segment_ids, input_mask)

        # slot_evaluate
        slot_output = slot_logits.argmax(dim=-1)
        slot_output = slot_output.tolist()

        slot_ids = slot_ids.tolist()
        s_preds.extend(slot_output)
        s_labels.extend(slot_ids)
        epoch_pbar.update(1)
    epoch_pbar.close()

    slot_preds = reconstruct_slot_bound(s_preds, s_labels, slot_label_list)
    sentences = [x.words for idx,x in enumerate(data_raw) ]
    slot_labels = [x.slots for x in data_raw]


    f1 = f1_score(slot_labels, slot_preds)
    acc = accuracy_score(slot_labels, slot_preds)
    # write_prediction_to_file(sentences, slot_preds, slot_labels)
    class_report_str = classification_report(slot_labels, slot_preds)
    class_report = reconstruct_class_report(class_report_str)
    eval_res = {"slot_acc":acc,"slot_f1": f1}
    print("%s dataset evaluate results: %s" %(mode, eval_res))
    return eval_res, class_report

def reconstruct_class_report(class_report_str):
    class_report = {}
    all_lines = class_report_str.split('\n')
    for line in all_lines[2:-4]:
        line = line.strip().split()
        class_report[line[0]] = [float(line[-2]), int(line[-1])]
    return class_report

def get_real_label(label_str):
    return label_str.split('-')[1] if '-' in label_str else 'O'

def reconstruct_slot_bound(slot_output, slot_ids, slot_label_list):
    # 1. 排除不需要的idx
    real_pred = []
    for p_list,l_list in zip(slot_output, slot_ids):
        single_pred = []
        for p,l in zip(p_list,l_list):
            if l == -100:
                continue
            single_pred.append(slot_label_list[p])
        real_pred.append(single_pred)
    # 2.恢复真实标签
    final_pred = []
    for single_pred in real_pred:
        final_single_pred = ['O'] if single_pred[0] == 'O' else ['B-'+single_pred[0]]
        for idx in range(1,len(single_pred)):
            cur_pred = single_pred[idx]
            pre_pred = single_pred[idx-1]
            if cur_pred == 'O':
                final_single_pred.append('O')
            else:
                if cur_pred == pre_pred:
                    final_single_pred.append('I-'+cur_pred)
                else:
                    final_single_pred.append('B-'+cur_pred)
        final_pred.append(final_single_pred)
    return final_pred


def write_prediction_to_file(sentences, slot_preds, slot_labels, filename='preds'):
    res = []
    bad = []
    cal_bad_case = [get_real_label(p)+'_to_'+get_real_label(l) for ps,ls in zip(slot_preds, slot_labels) for p,l in zip(ps,ls) if p !=l]
    for s,t,l in zip(sentences, slot_preds, slot_labels):
        res.append("input sentence: %s \n" % s)
        res.append("slot pred : %s \n" %" ".join(t))
        res.append("slot label: %s \n\n" %" ".join(l))
        if t != l:
            bad.append("input sentence: %s \n" % s)
            bad.append("slot pred : %s \n" % " ".join(t))
            bad.append("slot label: %s \n\n" % " ".join(l))

    with open(filename,'w',encoding='utf-8') as f:
        f.writelines(res)

    with open('bad_case','w',encoding='utf-8') as f:
        f.writelines(bad)

    with open('bad_case_detail.json','w',encoding='utf-8') as f:
        f.write(json.dumps(dict(Counter(cal_bad_case)),indent=2))

def cal_acc(preds, labels):
    acc = sum([1 if p == l else 0 for p, l in zip(preds, labels)]) / len(labels)
    return acc


def align_predictions(preds, slot_ids, id_to_label):
    aligned_labels = []
    aligned_preds = []
    for p, l in zip(preds, slot_ids):
        if l != -100:
            aligned_preds.append(id_to_label[p])
            aligned_labels.append(id_to_label[l])
    return aligned_preds, aligned_labels


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    # dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)

    prompt_sent_list = processor.get_prompt_sent(args.data_dir)
    label_inputs = construct_label_inputs(labels['slot_prompts'], tokenizer)
    # dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.use_crf = args.use_crf
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    model = Model.from_pretrained(config=model_config, pretrained_model_name_or_path=args.model_ckpt_path)


    # ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    # model.load_state_dict(ckpt, strict=False)
    model.to(device)
    # evaluate(model, dev_data_raw, labels,tokenizer, 'dev')
    evaluate(model, test_data_raw, labels,tokenizer, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/snips/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/temp/model_best.bin', type=str)
    parser.add_argument("--use_crf", default=False, type=bool)
    parser.add_argument("--max_seq_len", default=80, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()

    args.bert_ckpt_path = os.path.join(args.model_path, 'pytorch_model.bin')
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
