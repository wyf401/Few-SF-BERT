import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
import torch
from torch.autograd import Variable


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs

        slot_logits = self.slot_classifier(sequence_output)
        return slot_logits



class FocalLoss(nn.Module):
    def __init__(self, gamma=0., ignore_index=-100, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        num_label = input.size()[-1]

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask.unsqueeze(-1).expand_as(input)).view(-1, num_label)
        target = torch.masked_select(target, mask)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()