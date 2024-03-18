model_urls = {
    # all the checkpoints come from https://github.com/HRNet/HRNet-Image-Classification
    'hrnet18': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}
import torch.nn as nn
import torch
from networks.basic_blocks import *
BN_MOMENTUM = 0.1
import torch.nn.functional as F
import numpy as np
from networks.basic_blocks import *


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=1, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        # print(probs.shape,feats.shape)
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_parallel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.cmip = CMIP()

        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # print('conv1',out[1].shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.cmip(out, self.bn2_list)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_parallel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.cmip = CMIP()
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.cmip(out, self.bn2_list)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c, num_parallel=2):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                BasicBlock(w, w, num_parallel),
                BasicBlock(w, w, num_parallel),
                BasicBlock(w, w, num_parallel),
                BasicBlock(w, w, num_parallel)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block

        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        # 接着融合不同尺寸信息
        x_fused0 = []
        x_fused1 = []
        x_fused = []

        for i in range(len(self.fuse_layers)):
            x_fused0.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j][0]) for j in range(len(self.branches))])
                )
            )
        for i in range(len(self.fuse_layers)):
            x_fused1.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j][1]) for j in range(len(self.branches))])
                )
            )
        for x, y in zip(x_fused0, x_fused1):
            x_fused.append([x, y])

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 18, num_parallel=2):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.conv1_add = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_add = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2_add = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_add = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            ModuleParallel(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)),
            BatchNorm2dParallel(256, num_parallel)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, num_parallel, downsample=downsample),
            Bottleneck(256, 64, num_parallel),
            Bottleneck(256, 64, num_parallel),
            Bottleneck(256, 64, num_parallel)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                ModuleParallel(nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False)),
                BatchNorm2dParallel(base_channel, num_parallel),
                ModuleParallel(nn.ReLU(inplace=True))
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    ModuleParallel(nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False)),
                    BatchNorm2dParallel(base_channel * 2, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True))
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            ModuleParallel(nn.Identity()),  # None,  - Used in place of "None" because it is callable
            ModuleParallel(nn.Identity()),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    ModuleParallel(
                        nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False)),
                    BatchNorm2dParallel(base_channel * 4, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True))
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            ModuleParallel(nn.Identity()),  # None,  - Used in place of "None" because it is callable
            ModuleParallel(nn.Identity()),  # None,  - Used in place of "None" because it is callable
            ModuleParallel(nn.Identity()),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    ModuleParallel(
                        nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False)),
                    BatchNorm2dParallel(base_channel * 8, num_parallel),
                    ModuleParallel(nn.ReLU(inplace=True))
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel)
        )

        # Final layer

        MID_CHANNELS = 512
        KEY_CHANNELS = 256
        last_inp_channels = base_channel
        ocr_mid_channels = MID_CHANNELS
        ocr_key_channels = KEY_CHANNELS

        last_inp_channels = (1 + 2 + 4 + 8) * 2 * base_channel
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        # self.fuse3x3 = nn.Conv2d(2*last_inp_channels,last_inp_channels, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv3x3_ocr = nn.Sequential(
        #     ModuleParallel(nn.Conv2d(last_inp_channels, ocr_mid_channels,
        #               kernel_size=3, stride=1, padding=1)),
        #     BatchNorm2dParallel(ocr_mid_channels,num_parallel),
        #     ModuleParallel(nn.ReLU(inplace=True)),
        # )
        # self.ocr_gather_head = SpatialGather_Module(9)
        #
        # self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
        #                                          key_channels=ocr_key_channels,
        #                                          out_channels=ocr_mid_channels,
        #                                          scale=1,
        #                                          dropout=0.05,
        #                                          )
        # self.cls_head = nn.Conv2d(
        #     ocr_mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        # self.aux_head = nn.Sequential(
        #     ModuleParallel(nn.Conv2d(last_inp_channels, last_inp_channels,
        #               kernel_size=1, stride=1, padding=0)),
        #     BatchNorm2dParallel(last_inp_channels,num_parallel),
        #     ModuleParallel(nn.ReLU(inplace=True)),
        #     ModuleParallel(nn.Conv2d(last_inp_channels, 2,
        #               kernel_size=1, stride=1, padding=0, bias=True))
        # )

    def forward(self, inputs):
        x = inputs[:, :3, :, :]
        y = inputs[:, 3:, :, :]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        y = self.relu(self.bn1_add(self.conv1_add(y)))
        y = self.relu(self.bn2_add(self.conv2_add(y)))
        x = [x, y]
        x = self.layer1(x)

        x = [trans(x) for trans in self.transition1]  # Since now, x is a list
        x = self.stage2(x)

        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only
        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only
        x = self.stage4(x)
        x0_h, x0_w = x[0][0].size(2), x[0][0].size(3)

        ALIGN_CORNERS = True
        x10 = F.interpolate(x[1][0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x20 = F.interpolate(x[2][0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x30 = F.interpolate(x[3][0], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x11 = F.interpolate(x[1][1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x21 = F.interpolate(x[2][1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x31 = F.interpolate(x[3][1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        feats0 = torch.cat([x[0][0], x10, x20, x30], 1)
        feats1 = torch.cat([x[0][1], x11, x21, x31], 1)
        # feat=[feats0,feats1]
        # feats = self.conv3x3_ocr(feat)
        # out_aux = self.aux_head(feat)
        # context = self.ocr_gather_head(feats, out_aux)
        # feats = self.ocr_distri_head(feats, context)

        # out = self.cls_head(feats)

        feats = torch.cat([feats0, feats1], 1)

        out = self.last_layer(feats)
        out = F.interpolate(input=out, size=(
            inputs.shape[2], inputs.shape[3]), mode='bilinear', align_corners=True)
        out = torch.sigmoid(out)

        return out


def hrnet32(pretrained):
    pretrained_dict = torch.load(
        "/home/imi432004/hrnet/experiment/CMMPNet_lin/pre-trained_weights/hrnetv2_w32_imagenet_pretrained.pth")
    model = HighResolutionNet(base_channel=32)

    if pretrained:
        print('=> loading pretrained model {}'.format(pretrained))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        # for k, _ in pretrained_dict.items():
        # print('=> loading {} pretrained model {}'.format(k, pretrained))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict


def CMIP_hrnet18(pretrained):

    model = HighResolutionNet(base_channel=18)
    if pretrained:
        pretrained_dict = torch.load(
            'pre-trained_weights/hrnetv2_w18_imagenet_pretrained.pth')
        print('=> loading pretrained model {}'.format(pretrained))
        num_parallel = 2
        model_dict = expand_model_dict(model.state_dict(), pretrained_dict, num_parallel)
        model.load_state_dict(model_dict, strict=True)
    return model

# model = CMIP_hrnet18(pretrained=True)
# input = torch.randn([2, 4, 512, 512], dtype=torch.float)
# output = model(input)
# print(output.shape)