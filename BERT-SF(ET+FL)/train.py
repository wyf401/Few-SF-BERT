import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertConfig, AdamW
from transformers.trainer import get_linear_schedule_with_warmup

from data.utils import generate_fewshot_data_mj, readfile
from evaluate import evaluate
from model import Model, FocalLoss
from utils.ckpt_utils import download_ckpt
from utils.data_utils import prepare_data, NluDataset, glue_processor, construct_label_inputs


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    train_examples = processor.get_train_examples(args.train_data_path)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    prompt_sent_list = processor.get_prompt_sent(args.data_dir)
    label_inputs = construct_label_inputs(labels['slot_prompts'], tokenizer)
    num_train_example = len(train_examples)
    # [:int(num_train_example/100*5)]

    train_data_raw = prepare_data(train_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)
    dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)

    print("# train examples %d" % len(train_data_raw))
    print("# dev examples %d" % len(dev_data_raw))
    print("# test examples %d" % len(test_data_raw))
    train_data = NluDataset(train_data_raw)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  # len(labels["intent_labels"]),# args.batch_size,
                                  collate_fn=train_data.collate_fn, sampler=RandomSampler(train_data))

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.use_crf = args.use_crf
    model_config.max_seq_len = args.max_seq_len
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')
    model = Model.from_pretrained(config=model_config, pretrained_model_name_or_path=args.bert_ckpt_path)
    model.to(device)

    # Optimizer
    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_train_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * warmup),num_train_steps)
    loss_fnc = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # loss_fnc = FocalLoss(gamma=2)
    loss_fnc.to(device)
    # Training
    best_score = {"joint_acc": 0, "intent_acc": 0, "slot_acc": 0, "slot_precision": 0, "slot_recall": 0,
                  "slot_f1": 0}
    best_epoch = 0
    train_pbar = trange(0, args.n_epochs, desc="Epoch")
    for epoch in range(args.n_epochs):
        batch_loss = []
        epoch_pbar = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, segment_ids, input_mask, slot_ids = batch
            slot_logits = model(input_ids, segment_ids, input_mask)

            loss_slot = loss_fnc(slot_logits.view(-1, len(labels['slot_labels'])), slot_ids.view(-1))

            # loss_intent = loss_fnc(seq_relationship_score, intent_id)
            loss = loss_slot

            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            epoch_pbar.update(1)
            if (step + 1) % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" \
                      % (epoch + 1, args.n_epochs, step + 1,
                         len(train_dataloader), np.mean(batch_loss)))
        epoch_pbar.close()
        print('Epoch %d mean loss: %.3f' % (epoch + 1, np.mean(batch_loss)))
        res,_ = evaluate(model, dev_data_raw, labels, tokenizer)
        if res['slot_f1'] >= best_score['slot_f1']:
            best_score = res
            best_epoch = epoch + 1
            save_path = os.path.join(args.save_dir, 'model_best.bin')
            torch.save(model.state_dict(), save_path)
        print("Best Score : ", best_score, 'in epoch ', best_epoch)
        train_pbar.update(1)
    train_pbar.close()
    ckpt = torch.load(os.path.join(args.save_dir, 'model_best.bin'))
    model.load_state_dict(ckpt, strict=False)
    res,class_report = evaluate(model, test_data_raw, labels, tokenizer, mode="test")
    return res,class_report


def get_new_sample(label_to_num, last_res, iteration_percentage, add):
    # sample strategy
    # 1. 固定average增长量,　和iteration_percentage负相关
    beta = 1
    average_num = int((1 - iteration_percentage) ** beta * add)
    for key in last_res:
        label_to_num[key] += average_num
    # 2. 和last_res负相关的采样
    alpha = 1
    total_num_left = (add - average_num) * len(label_to_num)
    last_res_error = {k: ((1 - v[0])*(v[1]**0.3)) ** alpha for k, v in last_res.items()}
    total_error = sum(last_res_error.values())
    for key, error in last_res_error.items():
        label_to_num[key] += round(error / total_error * total_num_left)

    print('continue sampling according to last result: %s' %label_to_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/snips/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--use_crf", default=False, type=bool)
    parser.add_argument("--save_dir", default='outputs/temp', type=str)
    parser.add_argument("--max_seq_len", default=80, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--n_epochs", default=15, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_ckpt_path = os.path.join(args.model_path, 'pytorch_model.bin')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)

    num_sample = [5, 10, 20]
    ori_path = args.data_dir
    target_path = os.path.join(ori_path, 'dy_sample')
    slot_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_path, "vocab/slot_vocab"))]

    label_to_num = {x: num_sample[0] for x in slot_label_list}
    for idx, num_each_cls in enumerate(num_sample):
        print('%s-shot training starts ...' %str(num_each_cls))
        train_data_path = generate_fewshot_data_mj(ori_path, target_path, num_each_cls, label_to_num)
        args.train_data_path = train_data_path
        res,class_report = main(args)
        print('%s-shot training ends' % str(num_each_cls))
        print('final test res : %s' %res)
        print('classification_report: %s' %class_report)
