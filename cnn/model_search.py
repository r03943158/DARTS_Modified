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
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm

# parser = argparse.ArgumentParser("cifar")
# parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
# parser.add_argument('--batch_size', type=int, default=64, help='batch size')
# parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
# parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
# parser.add_argument('--save', type=str, default='EXP', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
# parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
# parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--num_samples', type=int, default=5, help='number of models to sample at one time')
# parser.add_argument('--child_epochs', type=int, default=10, help='number of epochs to train one path')
# args = parser.parse_args()
#
# CIFAR_CLASSES = 10
#
# def random_sample(cell_shape):
#     weights = torch.zeros(cell_shape)
#     for i in range(cell_shape[0]):
#         choice = np.random.randint(cell_shape[1])
#         weights[i, choice] = 1
#     weights = weights.float().cuda()
#     return weights
#
# def train_one_path(train_queue, valid_queue, model, criterion, optimizer):
#     for step, (input, target) in enumerate(train_queue):
#         model.train()
#         n = input.size(0)
#
#         input = Variable(input, requires_grad=False).cuda()
#         target = Variable(target, requires_grad=False).cuda()
#
#         optimizer.zero_grad()
#         logits = model(input)
#         loss = criterion(logits, target)
#
#         loss.backward()
#         nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
#         optimizer.step()
#
# def accumulate_gradient(train_queue, valid_queue, model, criterion, optimizer):
#     optimizer.zero_grad()
#     for step, (input, target) in enumerate(train_queue):
#         model.train()
#         n = input.size(0)
#
#         input = Variable(input, requires_grad=False).cuda()
#         target = Variable(target, requires_grad=False).cuda()
#
#         logits = model(input)
#         loss = criterion(logits, target)
#
#         loss.backward()
#
# def add_gradient(target_model, reference_model, beta):
#     for paramName, paramValue, in reference_model.named_parameters():
#         for netCopyName, netCopyValue, in target_model.named_parameters():
#             if paramName == netCopyName:
#                 netCopyValue.grad += beta * paramValue.grad.clone()
#
# def validate(valid_queue, model, criterion):
#     objs = utils.AvgrageMeter()
#     top1 = utils.AvgrageMeter()
#     top5 = utils.AvgrageMeter()
#     model.eval()
#
#     for step, (input, target) in enumerate(valid_queue):
#         input = Variable(input, volatile=True).cuda()
#         target = Variable(target, volatile=True).cuda()
#
#         logits = model(input)
#         loss = criterion(logits, target)
#
#         prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
#         n = input.size(0)
#         objs.update(loss.item(), n)
#         top1.update(prec1.item(), n)
#         top5.update(prec5.item(), n)
#
#     return top1.avg, objs.avg
#
# if __name__ == '__main__':
#     # criterion = nn.CrossEntropyLoss()
#     # model = Network(16, 10, 8, criterion)
#     # print(model.alphas_normal.shape)
#     # print(model.alphas_reduce.shape)
#     writer = SummaryWriter()
#     np.random.seed(args.seed)
#     torch.cuda.set_device(args.gpu)
#     #cudnn.benchmark = True
#     torch.manual_seed(args.seed)
#     #cudnn.enabled=True
#     torch.cuda.manual_seed(args.seed)
#
#     criterion = nn.CrossEntropyLoss()
#     criterion = criterion.cuda()
#
#     model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, None)
#     model = model.cuda()
#
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         args.learning_rate,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay)
#
#     train_transform, valid_transform = utils._data_transforms_cifar10(args)
#     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
#
#     num_train = len(train_data)
#     indices = list(range(num_train))
#     split = int(np.floor(args.train_portion * num_train))
#
#     train_queue = torch.utils.data.DataLoader(
#         train_data, batch_size=args.batch_size,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#         pin_memory=True, num_workers=2)
#
#     valid_queue = torch.utils.data.DataLoader(
#         train_data, batch_size=args.batch_size,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
#         pin_memory=True, num_workers=2)
#
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     #     optimizer, float(args.epochs), eta_min=args.learning_rate_min)
#
#     for epoch in range(args.epochs):
#         print("Current epoch: {}".format(epoch))
#         optimizer.zero_grad()
#         avg_acc, avg_loss = 0.0, 0.0
#         model_buffer = []
#         optimizer_buffer = []
#         for sample in range(args.num_samples):
#             cell_shape = (sum(1 for i in range(4) for n in range(2+i)), len(PRIMITIVES))
#             sampled_weights = random_sample(cell_shape)
#             model_buffer.append(Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, sampled_weights).cuda())
#             optimizer_buffer.append(torch.optim.SGD(model_buffer[-1].parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay))
#             for child_epoch in tqdm(range(args.child_epochs)):
#                 train_one_path(train_queue, valid_queue, model_buffer[-1], criterion, optimizer_buffer[-1])
#             acc, loss = validate(valid_queue, model_buffer[-1], criterion)
#             avg_acc += acc
#             avg_loss += loss
#
#             accumulate_gradient(train_queue, valid_queue, model_buffer[-1], criterion, optimizer_buffer[-1])
#             add_gradient(target_model=model, reference_model=model_buffer[-1], beta=1/len(train_queue))
#
#             del model_buffer[:]
#             del optimizer_buffer[:]
#         optimizer.step()
#
#         avg_acc /= args.num_samples
#         avg_loss /= args.num_samples
#
#         writer.add_scalar('avg_one_acc', avg_acc, epoch)
#         writer.add_scalar('avg_one_loss', avg_loss, epoch)
#
#     writer.close()

    # for epoch in range(args.epochs):
    #     scheduler.step()
    #     lr = scheduler.get_lr()[0]
    #
    #     # training
    #     train_acc, train_obj = train(train_queue, valid_queue, model, criterion, optimizer, lr, weights)
    #
    #     # validation
    #     valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #
    #     utils.save(model, os.path.join(args.save, 'weights.pt'))
