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

from child_model import *

def initialize_gradients(model, optimizer):
    optimizer.zero_grad()
    input = torch.ones(3, 3, 32, 32).float()
    output = model(input)
    loss = output.sum()
    loss.backward()
    optimizer.zero_grad()

def copyChildToParent(parent, child, weight_copy=False, grad_copy=False):
    assert len(parent.cells) == len(child.cells)

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.stem.named_parameters(), child.stem.named_parameters()):
        if weight_copy == True:
            paramValueParent.data = paramValueChild.data.clone()
        if grad_copy == True:
            paramValueParent.grad = paramValueChild.grad.clone()

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.global_pooling.named_parameters(), child.global_pooling.named_parameters()):
        if weight_copy == True:
            paramValueParent.data = paramValueChild.data.clone()
        if grad_copy == True:
            paramValueParent.grad = paramValueChild.grad.clone()

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.classifier.named_parameters(), child.classifier.named_parameters()):
        if weight_copy == True:
            paramValueParent.data = paramValueChild.data.clone()
        if grad_copy == True:
            paramValueParent.grad = paramValueChild.grad.clone()

    for i in range(len(parent.cells)):
        parent_cell = parent.cells[i]
        child_cell = child.cells[i]

        for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_cell.preprocess0.named_parameters(), child_cell.preprocess0.named_parameters()):
            if weight_copy == True:
                paramValueParent.data = paramValueChild.data.clone()
            if grad_copy == True:
                paramValueParent.grad = paramValueChild.grad.clone()

        for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_cell.preprocess1.named_parameters(), child_cell.preprocess1.named_parameters()):
            if weight_copy == True:
                paramValueParent.data = paramValueChild.data.clone()
            if grad_copy == True:
                paramValueParent.grad = paramValueChild.grad.clone()

        for j in range(len(parent_cell._ops)):
            parent_op = parent_cell._ops[j]
            child_op = child_cell._ops[j]
            choice_index = child_op.choice_index

            for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_op._ops[choice_index].named_parameters(), child_op._ops[0].named_parameters()):
                # print(paramNameParent)
                # print(paramNameChild)

                if weight_copy == True:
                    paramValueParent.data = paramValueChild.data.clone()
                if grad_copy == True:
                    paramValueParent.grad = paramValueChild.grad.clone()

def copyParentToChild(parent, child, weight_copy=False, grad_copy=False):
    assert len(parent.cells) == len(child.cells)

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.stem.named_parameters(), child.stem.named_parameters()):
        if weight_copy == True:
            paramValueChild.data = paramValueParent.data.clone()
        if grad_copy == True:
            paramValueChild.grad = paramValueParent.grad.clone()

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.global_pooling.named_parameters(), child.global_pooling.named_parameters()):
        if weight_copy == True:
            paramValueChild.data = paramValueParent.data.clone()
        if grad_copy == True:
            paramValueChild.grad = paramValueParent.grad.clone()

    for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent.classifier.named_parameters(), child.classifier.named_parameters()):
        if weight_copy == True:
            paramValueChild.data = paramValueParent.data.clone()
        if grad_copy == True:
            paramValueChild.grad = paramValueParent.grad.clone()

    for i in range(len(parent.cells)):
        parent_cell = parent.cells[i]
        child_cell = child.cells[i]

        for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_cell.preprocess0.named_parameters(), child_cell.preprocess0.named_parameters()):
            if weight_copy == True:
                paramValueChild.data = paramValueParent.data.clone()
            if grad_copy == True:
                paramValueChild.grad = paramValueParent.grad.clone()

        for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_cell.preprocess1.named_parameters(), child_cell.preprocess1.named_parameters()):
            if weight_copy == True:
                paramValueChild.data = paramValueParent.data.clone()
            if grad_copy == True:
                paramValueChild.grad = paramValueParent.grad.clone()

        for j in range(len(parent_cell._ops)):
            parent_op = parent_cell._ops[j]
            child_op = child_cell._ops[j]
            choice_index = child_op.choice_index

            for (paramNameParent, paramValueParent), (paramNameChild, paramValueChild) in zip(parent_op._ops[choice_index].named_parameters(), child_op._ops[0].named_parameters()):
                # print(paramNameParent)
                # print(paramNameChild)

                if weight_copy == True:
                    paramValueChild.data = paramValueParent.data.clone()

                if grad_copy == True:
                    paramValueChild.grad = paramValueParent.grad.clone()

class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x):
        return sum(op(x) for op in self._ops)

class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

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

class SuperNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, steps=4, multiplier=4, stem_multiplier=3):
        super(SuperNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            # if cell.reduction:
            #     weights = F.softmax(self.alphas_reduce, dim=-1)
            # else:
            #     weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

if __name__ == '__main__':
    num_layers = 1
    sampled_architecture = random_sample((num_layers, sum(1 for i in range(4) for n in range(2+i)))).astype(int)
    print(sampled_architecture)
    child_model = ChildNetwork(16, 10, num_layers, sampled_architecture)
    parent_model = SuperNetwork(16, 10, num_layers)

    parent_optimizer = torch.optim.SGD(
        parent_model.parameters(),
        0.001,
        momentum=0.9,
        weight_decay=3e-4)

    child_optimizer = torch.optim.SGD(
        child_model.parameters(),
        0.001,
        momentum=0.9,
        weight_decay=3e-4)

    initialize_gradients(child_model, child_optimizer)
    initialize_gradients(parent_model, parent_optimizer)
    copyParentToChild(parent_model, child_model, True, True)
    copyChildToParent(parent_model, child_model, True, True)
