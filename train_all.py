#coding=utf-8
import argparse
import os
import time
import logging
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

cudnn.benchmark = True

import numpy as np
import models

from data import datasets
from data.sampler import CycleSampler
from data.data_utils import init_fn
from utils import Parser,criterions

from predict import AverageMeter
import setproctitle  # pip install setproctitle

parser = argparse.ArgumentParser()

parser.add_argument('-cfg', '--cfg', default='1_EESPNet_16x_PRelu_GDL_all', required=True, type=str,
                    help='Your detailed configuration of the network')
parser.add_argument('-gpu', '--gpu', default='0', type=str, required=True,
                    help='Supprot one GPU & multiple GPUs.')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int,
                    help='Batch size')
parser.add_argument('-restore', '--restore', default='model_last.pth', type=str)# model_last.pth

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
args = Parser(args.cfg, log='train').add_args(args)
# args.net_params.device_ids= [int(x) for x in (args.gpu).split(',')]
ckpts = args.makedir()

args.resume = os.path.join(ckpts,args.restore) # specify the epoch

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    Network = getattr(models, args.net) #
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = getattr(torch.optim, args.opt)(model.parameters(), **args.opt_params)
    criterion = getattr(criterions, args.criterion)

    msg = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_dict'])
            msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
    else:
        msg = '-------------- New training session ----------------'

    msg += '\n' + str(args)
    logging.info(msg)

    # Data loading code
    Dataset = getattr(datasets, args.dataset) #

    train_list = os.path.join(args.train_data_dir, args.train_list)
    train_set = Dataset(train_list, root=args.train_data_dir, for_train=True,transforms=args.train_transforms)

    num_iters = args.num_iters or (len(train_set) * args.num_epochs) // args.batch_size
    num_iters -= args.start_iter
    train_sampler = CycleSampler(len(train_set), num_iters*args.batch_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=train_set.collate,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=init_fn)

    start = time.time()

    enum_batches = len(train_set)/ float(args.batch_size) # nums_batch per epoch

    losses = AverageMeter()
    torch.set_grad_enabled(True)

    for i, data in enumerate(train_loader, args.start_iter):

        elapsed_bsize = int( i / enum_batches)+1
        epoch = int((i + 1) / enum_batches)
        setproctitle.setproctitle("Epoch:{}/{}".format(elapsed_bsize,args.num_epochs))

        # actual training
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.opt_params.lr)

        data = [t.cuda(non_blocking=True) for t in data]
        x, target = data[:2]

        output = model(x)

        if not args.weight_type: # compatible for the old version
            args.weight_type = 'square'

        if args.criterion_kwargs is not None:
            loss = criterion(output, target, **args.criterion_kwargs)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), target.numel())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % int(enum_batches * args.save_freq) == 0 \
            or (i+1) % int(enum_batches * (args.num_epochs -1))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -2))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -3))==0\
            or (i+1) % int(enum_batches * (args.num_epochs -4))==0:

            file_name = os.path.join(ckpts, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'iter': i,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)


        msg = 'Iter {0:}, Epoch {1:.4f}, Loss {2:.7f}'.format(i+1, (i+1)/enum_batches, losses.avg)
        logging.info(msg)

        losses.reset()

    i = num_iters + args.start_iter
    file_name = os.path.join(ckpts, 'model_last.pth')
    torch.save({
        'iter': i,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
        },
        file_name)

    msg = 'total time: {:.4f} minutes'.format((time.time() - start)/60)
    logging.info(msg)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)


if __name__ == '__main__':
    main()
