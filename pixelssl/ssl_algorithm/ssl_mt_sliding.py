import os
import time
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pixelssl.utils import REGRESSION, CLASSIFICATION
from pixelssl.utils import logger, cmd, tool
from pixelssl.nn import func
from pixelssl.nn.module import patch_replication_callback, GaussianNoiseLayer

from ssl_mt import SSLMT

from . import ssl_base


""" Implementation of pixel-wise Mean Teacher (MT)

This method is proposed in the paper:
    'Mean Teachers are Better Role Models:
        Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results'

This implementation only supports Gaussian noise as input perturbation, and the two-heads
outputs trick is not available.
"""


def add_parser_arguments(parser):
    ssl_base.add_parser_arguments(parser)
    parser.add_argument('--cons-for-labeled', type=cmd.str2bool, default=True, help='sslmt - calculate the consistency constraint on the labeled data if True')
    parser.add_argument('--cons-scale', type=float, default=-1, help='sslmt - consistency constraint coefficient')
    parser.add_argument('--cons-rampup-epochs', type=int, default=-1, help='sslmt - ramp-up epochs of conistency constraint')

    parser.add_argument('--ema-decay', type=float, default=0.999, help='sslmt - EMA coefficient of teacher model')
    parser.add_argument('--gaussian-noise-std', type=float, default=None, help='sslmt - std of input gaussian noise (set to None to disable it)')

class SSLMT_SLIDING(SSLMT):
    def _train(self, data_loader, epoch):
        self.meters.reset()
        lbs = self.args.labeled_batch_size

        self.s_model.train()
        self.t_model.train()

        for idx, (inp, gt) in enumerate(data_loader):
            timer = time.time()

            # 's_inp', 't_inp' and 'gt' are tuples
            s_inp, t_inp, gt = self._batch_prehandle(inp, gt, True)
            if len(gt) > 1 and idx == 0:
                self._inp_warn()

            # calculate the ramp-up coefficient of the consistency constraint
            cur_step = len(data_loader) * epoch + idx
            total_steps = len(data_loader) * self.args.cons_rampup_epochs
            cons_rampup_scale = func.sigmoid_rampup(cur_step, total_steps)

            self.s_optimizer.zero_grad()

            # forward the student model
            s_resulter, s_debugger = self.s_model.forward(s_inp)
            if not 'pred' in s_resulter.keys() or not 'activated_pred' in s_resulter.keys():
                self._pred_err()
            s_pred = tool.dict_value(s_resulter, 'pred')
            s_activated_pred = tool.dict_value(s_resulter, 'activated_pred')

            # calculate the supervised task constraint on the labeled data
            l_s_pred = func.split_tensor_tuple(s_pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_s_inp = func.split_tensor_tuple(s_inp, 0, lbs)

            # 'task_loss' is a tensor of 1-dim & n elements, where n == batch_size
            s_task_loss = self.s_criterion.forward(l_s_pred, l_gt, l_s_inp)
            s_task_loss = torch.mean(s_task_loss)
            self.meters.update('s_task_loss', s_task_loss.data)

            # forward the teacher model
            with torch.no_grad():
                t_inp = make_sliding_batches(t_inp)
                t_resulter, t_debugger = self.t_model.forward(t_inp)
                if not 'pred' in t_resulter.keys():
                    self._pred_err()
                t_pred = tool.dict_value(t_resulter, 'pred')
                t_activated_pred = tool.dict_value(t_resulter, 'activated_pred')

                # calculate 't_task_loss' for recording
                l_t_pred = func.split_tensor_tuple(t_pred, 0, lbs)
                l_t_inp = func.split_tensor_tuple(t_inp, 0, lbs)
                t_task_loss = self.s_criterion.forward(l_t_pred, l_gt, l_t_inp)
                t_task_loss = torch.mean(t_task_loss)
                self.meters.update('t_task_loss', t_task_loss.data)

            # calculate the consistency constraint from the teacher model to the student model
            t_pred = unslide_batches(t_pred)
            t_pseudo_gt = Variable(t_pred[0].detach().data, requires_grad=False)

            if self.args.cons_for_labeled:
                cons_loss = self.cons_criterion(s_pred[0], t_pseudo_gt)
            elif self.args.unlabeled_batch_size > 0:
                cons_loss = self.cons_criterion(s_pred[0][lbs:, ...], t_pseudo_gt[lbs:, ...])
            else:
                cons_loss = self.zero_tensor
            cons_loss = cons_rampup_scale * self.args.cons_scale * torch.mean(cons_loss)
            self.meters.update('cons_loss', cons_loss.data)

            # backward and update the student model
            loss = s_task_loss + cons_loss
            loss.backward()
            self.s_optimizer.step()

            # update the teacher model by EMA
            self._update_ema_variables(self.s_model, self.t_model, self.args.ema_decay, cur_step)

            # logging
            self.meters.update('batch_time', time.time() - timer)
            if idx % self.args.log_freq == 0:
                logger.log_info('step: [{0}][{1}/{2}]\tbatch-time: {meters[batch_time]:.3f}\n'
                                '  student-{3}\t=>\t'
                                's-task-loss: {meters[s_task_loss]:.6f}\t'
                                's-cons-loss: {meters[cons_loss]:.6f}\n'
                                '  teacher-{3}\t=>\t'
                                't-task-loss: {meters[t_task_loss]:.6f}\n'
                                .format(epoch + 1, idx, len(data_loader), self.args.task, meters=self.meters))

            # visualization
            if self.args.visualize and idx % self.args.visual_freq == 0:
                self._visualize(epoch, idx, True,
                                func.split_tensor_tuple(s_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(s_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_inp, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(t_activated_pred, 0, 1, reduce_dim=True),
                                func.split_tensor_tuple(gt, 0, 1, reduce_dim=True))

            # update iteration-based lrers
            if not self.args.is_epoch_lrer:
                self.s_lrer.step()

        # update epoch-based lrers
        if self.args.is_epoch_lrer:
            self.s_lrer.step()


def ssl_mt_sliding(args, model_dict, optimizer_dict, lrer_dict, criterion_dict, task_func):
    if not len(model_dict) == len(optimizer_dict) == len(lrer_dict) == len(criterion_dict) == 1:
        logger.log_err('The len(element_dict) of SSL_MT should be 1\n')
    elif list(model_dict.keys())[0] != 'model':
        logger.log_err('In SSL_MT, the key of element_dict should be \'model\',\n'
                       'but \'{0}\' is given\n'.format(model_dict.keys()))

    model_funcs = [model_dict['model']]
    optimizer_funcs = [optimizer_dict['model']]
    lrer_funcs = [lrer_dict['model']]
    criterion_funcs = [criterion_dict['model']]

    algorithm = SSLMT_SLIDING(args)
    algorithm.build(model_funcs, optimizer_funcs, lrer_funcs, criterion_funcs, task_func)
    return algorithm

def make_sliding_batches(t_inp):


