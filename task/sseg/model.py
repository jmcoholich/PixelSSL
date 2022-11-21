import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import pixelssl

from module import deeplab_v2, _pspnet

from torchvision.utils import save_image

def labelMapToIm(label, label_map):
    """
    label: H, W, 1 tensor
    label_map: [[[r, g, b], index]]...]
    """
    output = label.repeat(1, 1, 3)
    for color, id in label_map:
        output[label.squeeze(2)==id] = torch.tensor(color)

    return output

def add_parser_arguments(parser):
    pixelssl.model_template.add_parser_arguments(parser)

    # arguments for DeepLab
    parser.add_argument('--sliding-window-eval', type=pixelssl.str2bool, default=False, help='activate sliding window eval')
    parser.add_argument('--output-stride', type=int, default=16, help='sseg - output stride of the ResNet backbone')
    parser.add_argument('--backbone', type=str, default='resnet101', help='sseg - architecture of the backbone network')
    parser.add_argument('--freeze-bn', type=pixelssl.str2bool, default=False,
                        help='sseg - if true, the statistics in BatchNorm will not be updated')


def deeplabv2():
    return DeepLabV2


def pspnet():
    return PSPNet


class DeepLab(pixelssl.model_template.TaskModel):
    def __init__(self, args, version, pretrained_backbone_url=None):
        super(DeepLab, self).__init__(args)

        model_func = None
        if version == 'v2':
            model_func = deeplab_v2.DeepLabV2
        else:
            pixelssl.log_err('For Semantic Segmentation - DeepLab, '
                             'we do not support version: {0}\n'.format(version))

        self.model = model_func(backbone=self.args.backbone,
            output_stride=self.args.output_stride, num_classes=self.args.num_classes,
            sync_bn=True, freeze_bn=self.args.freeze_bn,
            pretrained_backbone_url=pretrained_backbone_url)

        self.param_groups = [
            {'params': self.model.get_1x_lr_params(), 'lr': self.args.lr},
            {'params': self.model.get_10x_lr_params(), 'lr': self.args.lr * 10}
        ]

    def make_sliding_windows(self, inp):
        # This is the sliding window eval implementation
        # print(data_no_aug.shape)

        # img1 = inp[0]
        # save_image(img1, 'pics/before_sliding.png')

        self.batch, self.channels, self.rows, self.columns = inp.shape
        self.output_classes = 21
        self.kernel_size = self.args.im_size
        # des_stride = 25
        # # import pdb; pdb.set_trace()
        # n_windows_row = max(int((self.rows - des_stride) / (self.kernel_size - des_stride)) + 1, 2)
        # n_windows_col = max(int((self.columns - des_stride) / (self.kernel_size - des_stride)) + 1, 2)
        # self.stride = (int((n_windows_row * self.kernel_size - self.rows)
        #                    / (n_windows_row - 1)),
        #                int((n_windows_col * self.kernel_size - self.rows)
        #                    / (n_windows_col - 1)))
        # if self.rows < self.kernel_size:
        #     padding_rows = int((self.kernel_size - self.rows) / 2) + 1
        # else:
        #     padding_rows = 0
        # if self.columns < self.kernel_size:
        #     padding_cols = int((self.kernel_size - self.columns) / 2) + 1
        # else:
        #     padding_cols = 0
        # self.padding = (padding_rows, padding_cols)
        self.padding = int(self.kernel_size // 3)
        self.stride = int(self.kernel_size // 10)
        inp = nn.functional.unfold(
            inp,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride)
        self.num_windows = inp.shape[-1]
        inp = inp.permute((0, 2, 1))
        inp = inp.reshape(self.batch * self.num_windows, self.channels, self.kernel_size, self.kernel_size)

        # for i in range(inp.shape[0]):
        #     img1 = inp[i]
        #     save_image(img1, f'pics/after_sliding_{i}.png')
        return inp

    def unmake_sliding_windows(self, pred, latent):

        # for i in range(pred.shape[0]):
        #     img1 = pred[i].argmax(dim=0).to(torch.float32).cpu()/21
        #     save_image(img1, f'pics/pred_before_fold_{i}.png')
        # print("teacher output[0]:", teacher_output_0.shape)
        pred = pred.reshape(self.batch, self.num_windows, self.output_classes * self.kernel_size**2)
        pred = pred.permute((0, 2, 1))
        temp = torch.ones_like(pred)
        pred = nn.functional.fold(
            pred,
            output_size=(self.rows, self.columns),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride)
        temp = nn.functional.fold(
            temp,
            output_size=(self.rows, self.columns),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride)
        # print("temp shape: ", temp.shape)
        # temp = temp[:, 0].to(torch.float32).cpu()
        # pre_conv_temp = pre_conv_temp.permute((1, 2, 0))

        # post_conv_temp = labelMapToIm(pre_conv_temp, [
        #     [[0, 0, 0], 0],
        #     [[256, 0, 0], 1],
        #     [[0, 256, 0], 2],
        #     [[0, 0, 256], 3],
        # ])
        # print("unique vals: ", temp.unique())
        # save_image(post_conv_temp, "pics/temp.png")
        # exit()

        # img1 = pred[0].argmax(dim=0).to(torch.float32).cpu()/21
        # save_image(img1, 'pics/pred_after_fold.png')
        # exit()
        # return pred, latent
        return pred / temp, latent

    def forward(self, inp, slide=False):
        resulter, debugger = {}, {}

        if not len(inp) == 1:
            pixelssl.log_err(
                'Semantic segmentation model DeepLab requires only one input\n'
                'However, {0} inputs are given\n'.format(len(inp)))
        inp = inp[0]
        if not self.training and self.sliding_window_eval:
            inp = self.make_sliding_windows(inp)

        pred, latent = self.model.forward(inp)

        if not self.training and self.sliding_window_eval:
            pred, latent = self.unmake_sliding_windows(pred, latent)
        resulter['pred'] = (pred, )
        resulter['activated_pred'] = (F.softmax(pred, dim=1), )
        resulter['ssls4l_rc_inp'] = pred
        resulter['sslcct_ad_inp'] = latent
        return resulter, debugger


class DeepLabV2(DeepLab):
    def __init__(self, args):

        if args.backbone == 'resnet50':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        elif args.backbone == 'resnet101':
            pretrained_backbone_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        elif args.backbone == 'resnet101-coco':
            pretrained_backbone_url = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        else:
            pixelssl.log_err('DeepLabV2 does not support the backbone: {0}\n'
                             'You can support it for DeepLabV2 in the file \'task/sseg/model.py\'\n'.format(args.backbone))

        super(DeepLabV2, self).__init__(args, 'v2', pretrained_backbone_url)


class PSPNet(pixelssl.model_template.TaskModel):
    def __init__(self, args):
        super(PSPNet, self).__init__(args)

        if self.args.backbone == 'resnet50':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        elif self.args.backbone == 'resnet101':
            self.pretrained_backbone_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
        elif args.backbone == 'resnet101-coco':
            self.pretrained_backbone_url = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        else:
            pixelssl.log_err('PSPNet does not support the backbone: {0}\n'
                             'You can support it for PSPNet in the file \'task/sseg/model.py\'\n'.format(args.backbone))

        self.model = _pspnet._PSPNet(backbone=self.args.backbone,
            output_stride=self.args.output_stride, num_classes=self.args.num_classes,
            sync_bn=True, freeze_bn=self.args.freeze_bn,
            pretrained_backbone_url=self.pretrained_backbone_url)

        self.param_groups = [
            {'params': filter(lambda p:p.requires_grad, self.model.get_backbone_params()), 'lr': self.args.lr},
            {'params': filter(lambda p:p.requires_grad, self.model.get_psp_params()), 'lr': self.args.lr * 10},
            {'params': filter(lambda p:p.requires_grad, self.model.get_decoder_params()), 'lr': self.args.lr * 10},
        ]

    def forward(self, inp):
        resulter, debugger = {}, {}

        if not len(inp) == 1:
            pixelssl.log_err(
                'Semantic segmentation model PSPNet requires only one input\n'
                'However, {0} inputs are given\n'.format(len(inp)))

        inp = inp[0]
        pred, latent = self.model.forward(inp)

        resulter['pred'] = (pred, )
        resulter['activated_pred'] = (F.softmax(pred, dim=1), )
        resulter['ssls4l_rc_inp'] = pred
        resulter['sslcct_ad_inp'] = latent

        return resulter, debugger
