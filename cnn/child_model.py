import torchvision.datasets as dset
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

class MixedOpSingle(nn.Module):
    def __init__(self, C, stride, choice_index):
        super(MixedOpSingle, self).__init__()
        self._ops = nn.ModuleList()
        self.choice_index = choice_index

        op = OPS[PRIMITIVES[choice_index]](C, stride, False)
        if 'pool' in PRIMITIVES[choice_index]:
            op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))

        # print(choice_index)
        # for paramName, paramValue, in op.named_parameters():
        #     print(PRIMITIVES[choice_index])
        #     print(paramName)
        # print()

        self._ops.append(op)

    def forward(self, x):
        return self._ops[0](x)

class CellSingle(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, choices):
        super(CellSingle, self).__init__()
        self.reduction = reduction
        self.choices = choices

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        counter = 0
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOpSingle(C, stride, choices[counter])
                self._ops.append(op)
                counter += 1

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

class ChildNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, choices, steps=4, multiplier=4, stem_multiplier=3):
        super(ChildNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self._choices = choices

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = CellSingle(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self._choices[i])
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

def random_sample(cell_shape):
    architecture = np.zeros(cell_shape)
    for i in range(cell_shape[0]):
        for j in range(cell_shape[1]):
            choice = np.random.randint(len(PRIMITIVES))
            architecture[i, j] = choice
    return architecture

if __name__ == '__main__':
    sampled_architecture = random_sample((8, sum(1 for i in range(4) for n in range(2+i)))).astype(int)
    print(sampled_architecture)
    model = ChildNetwork(16, 10, 8, sampled_architecture)
    # print(model.cells[0])
    # for paramName, paramValue, in model.named_parameters():
    #     print(paramName)
