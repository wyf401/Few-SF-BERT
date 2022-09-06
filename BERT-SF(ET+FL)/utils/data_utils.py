import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, logging

logger = logging.get_logger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class InputExample:
    guid: str
    words: List[str]
    intent: Optional[str]
    slots: Optional[List[str]]


class NluProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train/intent_seq.in")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "intent_seq.out")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "dev/intent_seq.out")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "predict")

    def get_labels(self, data_dir):
        """See base class."""
        slot_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/slot_vocab"))
        slot_labels = [label[0].split('=-=')[0] for label in slot_labels_list]
        slot_prompts = [label[0].split('=-=')[1] for label in slot_labels_list]
        intent_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/intent_vocab"))
        intent_labels = [label[0].split('=-=')[0] for label in intent_labels_list]

        labels = {'intent_labels': intent_labels, 'slot_labels': slot_labels, 'slot_prompts': slot_prompts}
        return labels

    def read_file(self, data_dir):
        with open(data_dir, 'r', encoding='utf-8') as f:
            res = f.readlines()
        return [x.strip() for x in res]

    def get_prompt_sent(self, data_dir):
        prompt_sent = self.read_file(os.path.join(data_dir, "vocab/sentence_prompt"))
        prompt_sent_A = prompt_sent[0].replace('[A]', '').strip()
        prompt_sent_B = prompt_sent[1].replace('[B]', '').strip()
        return [prompt_sent_A, prompt_sent_B]

    def _create_examples(self, lines_in, lines_out, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i, (line, out) in enumerate(zip(lines_in, lines_out)):
            label_split = out[0].strip().split()
            guid = "%s-%s" % (set_type, i)
            words = line[0][4:].strip()
            slots = None if set_type == "predict" else label_split[1:]
            intent = None if set_type == "predict" else label_split[0]
            examples.append(InputExample(guid=guid, words=words, slots=slots, intent=intent))

        return examples


class TrainingInstance:
    def __init__(self, example, max_seq_len):
        self.words = example.words
        self.slots = example.slots
        self.intent = example.intent
        self.max_seq_len = max_seq_len

    def make_instance(self, tokenizer, intent_label_map, slot_label_map, slot_prompts, pad_label_id=-100):
        # TODO 判断长度越界
        slot_label_map['[PAD]'] = pad_label_id

        word_tokens = []
        slot_ids = []
        if self.slots:
            assert len(self.words.split()) == len(self.slots)
            for word, label_ori in zip(self.words.split(), self.slots):
                label = label_ori.split('-')[1] if '-' in label_ori else label_ori
                single_word_tokens = tokenizer.tokenize(word)
                if len(single_word_tokens) > 0:
                    word_tokens.extend(single_word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    slot_ids.extend([slot_label_map[label]] + [pad_label_id] * (len(single_word_tokens) - 1))
        else:
            # 预测时，把需要预测的位置置为１
            # TODO: 修改预测时的处理
            for word in self.words:
                single_word_tokens = tokenizer.tokenize(word)
                if len(single_word_tokens) > 0:
                    word_tokens.extend(single_word_tokens)
                    slot_ids.extend([1] + [0] * (len(single_word_tokens) - 1))

        assert len(slot_ids) == len(word_tokens)

        # convert token to ids
        tokens = ["[CLS]"] + word_tokens + ["[SEP]"]
        assert len(tokens) <= self.max_seq_len
        self.input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.segment_id = [0] * len(self.input_ids)
        self.input_mask = [1] * len(self.input_ids)
        padding_length = self.max_seq_len - len(self.input_ids)
        if padding_length > 0:
            self.input_ids = self.input_ids + [0] * padding_length
            self.segment_id = self.segment_id + [0] * padding_length
            self.input_mask = self.input_mask + [0] * padding_length
            self.slot_ids = [-100] + slot_ids + [-100] * (padding_length + 1)

        self.prompt_input = " ".join(tokens + ["PAD"] * padding_length)


class NluDataset(Dataset):
    def __init__(self, data, annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_ids for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        # intent_id = torch.tensor([f.intent_id for f in batch], dtype=torch.float) if self.annotated else None
        slot_ids = torch.tensor([f.slot_ids for f in batch], dtype=torch.long)
        return input_ids, segment_ids, input_mask, slot_ids


def prepare_data(examples, max_seq_len, tokenizer, labels, prompt_sent_list):
    slot_label_map = {label: idx for idx, label in enumerate(labels['slot_labels'])}
    intent_label_map = {label: idx for idx, label in enumerate(labels['intent_labels'])}
    slot_prompts = labels['slot_prompts']
    data = []

    for idx, example in enumerate(examples):
        instance = TrainingInstance(example, max_seq_len)
        instance.make_instance(tokenizer, intent_label_map, slot_label_map, slot_prompts)
        if idx < -1:
            print('Training Example %s :' % idx)
            print("Input sentence: %s" % instance.prompt_input)
            print('Input_ids: %s' % instance.input_ids)
            print("Input segment ids: %s" % instance.segment_id)
            print('Slot label: %s ' % (instance.slot_ids))
        data.append(instance)
    return data


def construct_label_inputs(label_prompts, tokenizer, max_seq_len=20):
    tokens_list = [tokenizer.tokenize("[CLS] " + x + " [SEP]") for x in label_prompts]
    i_list = []
    s_list = []
    m_list = []
    for tokens in tokens_list:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_id = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        padding_length = max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            segment_id = segment_id + [0] * padding_length
            input_mask = input_mask + [0] * padding_length
        i_list.append(input_ids)
        s_list.append(segment_id)
        m_list.append(input_mask)
    input_ids = torch.tensor([x for x in i_list], dtype=torch.long).to(device)
    segment_ids = torch.tensor([x for x in s_list], dtype=torch.long).to(device)
    input_mask = torch.tensor([x for x in m_list], dtype=torch.long).to(device)
    res = {"input_ids":input_ids, "token_type_ids":segment_ids, "attention_mask":input_mask}
    return res


glue_processor = {
    'nlu': NluProcessor()
}
