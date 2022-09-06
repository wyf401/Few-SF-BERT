import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=0., ignore_index=-100,  alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        num_label = input.size()[-1]

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask.unsqueeze(-1).expand_as(input)).view(-1,num_label)
        target = torch.masked_select(target, mask)

        target = target.view(-1,1)
        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

out = torch.tensor([[1,2,3],[4,5,6],[1,2,7],[1,2,3]],dtype=torch.float32)
target = torch.tensor([1,2,2,-100],dtype=torch.long)
ce = torch.nn.CrossEntropyLoss()
loss_1 = ce(out,target)

fc = FocalLoss(gamma=0.5)
loss_2 = fc(out,target)
print(loss_1.item())
print(loss_2.item())

print(1- 0.8 ** 0.5)
print(900**0.3)
print(30**0.3)