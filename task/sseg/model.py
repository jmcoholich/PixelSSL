import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import pixelssl

from module import deeplab_v2, _pspnet

from torchvision.utils import save_image



def add_parser_arguments(parser):
    pixelssl.model_template.add_parser_arguments(parser)

    # arguments for DeepLab
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
        # num_windows = 6
        # assert rows == columns  # TODO this should be case once we fix the preprocessor to apply to same crop to the teacher image
        # self.kernel_size = 321 # TODO pass all of these in from the config files
        self.kernel_size = 321 # TODO pass all of these in from the config files
        self.padding = 300 # TODO
        self.stride = 50
        # img is 769x769   # TODO will I have to upsample after sliding? If not, have we considering trying that?
        inp = nn.functional.unfold(
            inp,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride) # TODO hard pytroch
        # print("after unfold:", inp.shape)
        self.num_windows = inp.shape[-1]
        inp = inp.permute((0, 2, 1))
        # print("after permute:", inp.shape)
        inp = inp.reshape(self.batch * self.num_windows, self.channels, self.kernel_size, self.kernel_size)

        # for i in range(inp.shape[0]):
        #     img1 = inp[i]
        #     save_image(img1, f'pics/after_sliding_{i}.png')


        # print("after reshape:", inp.shape)
        return inp

    def unmake_sliding_windows(self, pred, latent):

        # for i in range(pred.shape[0]):
        #     img1 = pred[i].argmax(dim=0).to(torch.float32).cpu()/21
        #     save_image(img1, f'pics/pred_before_fold_{i}.png')
        # print("teacher output[0]:", teacher_output_0.shape)
        pred = pred.reshape(self.batch, self.num_windows, self.output_classes * self.kernel_size**2)
        # print("teacher output[0] reshape:", pred.shape)
        pred = pred.permute((0, 2, 1))
        # print("teacher output[0] permute:", pred.shape)
        pred = nn.functional.fold(
            pred,
            output_size=(self.rows, self.columns),
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride)

        # img1 = pred[0].argmax(dim=0).to(torch.float32).cpu()/21
        # save_image(img1, 'pics/pred_after_fold.png')
        # exit()
        return pred, latent

    def forward(self, inp, slide=False):
        resulter, debugger = {}, {}

        if not len(inp) == 1:
            pixelssl.log_err(
                'Semantic segmentation model DeepLab requires only one input\n'
                'However, {0} inputs are given\n'.format(len(inp)))

        inp = inp[0]
        # if slide:
        if True:  #TODO
            inp = self.make_sliding_windows(inp)
            # import pdb; pdb.set_trace()
        pred, latent = self.model.forward(inp)
        if True:
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
